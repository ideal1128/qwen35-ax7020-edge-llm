// tb_gemv_core.v -- independent golden-model regression for gemv_core.
//
// The reference below is NOT a transcription of the RTL.  It evaluates the same
// expression the software engine uses (qwen_kernels.c bfp8_gemv, pl_apply.c
// pl_apply):
//
//     y[m] = sum_b (double)dots[m][b] * ldexp(1.0, exps[m][b]-134) * xscale[b]
//
// accumulated in double, with
//     dots[m][b] = sum_{i<64} w_int8 * x_int16          (exact integer)
//     xscale[b]  = amax_b / 16384                        (float32 on the wire)
//
// The RTL emits the fixed-point word W = floor(y * 2^20) clamped to int32, so
// the comparison is  W / 2^20  against  y_ref  with a tolerance that reflects
// the RTL's fixed-point quantisation, and the max absolute / relative error is
// reported rather than just pass/fail.
//
// Stimulus amplitude is auto-scaled so the golden y lands near +-1.2, which is
// the regime the real model runs in (|y| mean 0.77, max 1.82) and keeps the
// comparison away from the output saturation rails.
//
// Covered:
//   * dequant_en = 0  legacy raw int32 block dots, compared bit-exactly
//   * dequant_en = 1  realistic exponents (118..130), per-block xscale spread
//                     of 16x and of 1024x across the blocks of one row
//   * dequant_en = 1  xs_beats = 0  identity xscale path
//   * dequant_en = 1  saturation: exponent far above the representable range
//                     must clamp to an int32 rail, not wrap and not drop the block
//   * dequant_en = 1  flush: exponent far below the range must give ~0
//   * sticky dbg error bits must stay clear on every legal burst
`timescale 1ns / 1ps

module tb_gemv_core;

    localparam integer K_MAX  = 1024;
    localparam integer M_MAX  = 8;
    localparam integer BLOCK  = 64;
    localparam integer NB_MAX = K_MAX / BLOCK;     // 16

    // ---------------------------------------------------------------- clock / DUT
    reg clk = 1'b0;
    reg rstn = 1'b0;
    always #5 clk = ~clk;

    reg  [127:0] s_axis_tdata  = 128'd0;
    reg          s_axis_tvalid = 1'b0;
    wire         s_axis_tready;
    reg          s_axis_tlast  = 1'b0;

    wire [31:0]  m_axis_tdata;
    wire         m_axis_tvalid;
    wire         m_axis_tlast;
    reg          m_axis_tready = 1'b1;
    wire [31:0]  dbg;

    gemv_core #(
        .ELEM_PER_BEAT(16), .BLOCK(64), .XW(8), .X_DEPTH(256)
    ) dut (
        .aclk(clk), .aresetn(rstn), .dbg_rst(1'b0), .dbg(dbg),
        .s_axis_tdata(s_axis_tdata), .s_axis_tkeep(16'hFFFF),
        .s_axis_tvalid(s_axis_tvalid), .s_axis_tready(s_axis_tready), .s_axis_tlast(s_axis_tlast),
        .m_axis_tdata(m_axis_tdata), .m_axis_tvalid(m_axis_tvalid), .m_axis_tready(m_axis_tready),
        .m_axis_tlast(m_axis_tlast)
    );

    // dbg sticky bits
    wire dbg_err_arith  = dbg[10];
    wire dbg_err_tlast  = dbg[11];
    wire dbg_err_rowbnd = dbg[12];

    // ---------------------------------------------------------------- stimulus
    reg signed [15:0] xq [0:K_MAX-1];
    reg signed [7:0]  wq [0:M_MAX*K_MAX-1];
    reg        [7:0]  we [0:M_MAX*NB_MAX-1];
    reg        [31:0] xsf[0:NB_MAX-1];        // float32 image of xscale (what goes on the wire)

    reg signed [31:0] got_out [0:M_MAX*NB_MAX-1];
    integer nrecv = 0;
    integer nlast = 0;
    integer errors = 0;
    integer i, m, b, j;
    integer k_curr;
    integer seed;
    reg [127:0] beat;

    // precomputed power-of-two tables: no nested function calls inside the model
    real expsc [0:255];       // expsc[e] = 2^(e-134)
    real p2e   [0:255];       // p2e[eb]  = 2^(eb-127)
    real xs_dec[0:NB_MAX-1];  // decoded xscale used by the model
    real xs_rel[0:NB_MAX-1];  // per-block relative amplitude

    // worst-case error tracking over all non-saturating dequant cases
    real err_max_abs = 0.0;
    real err_max_rel = 0.0;
    real y_abs_max   = 0.0;
    integer err_ncmp = 0;

    always @(posedge clk) begin
        if (m_axis_tvalid && m_axis_tready) begin
            got_out[nrecv] = $signed(m_axis_tdata);
            nrecv = nrecv + 1;
            if (m_axis_tlast === 1'b1) nlast = nlast + 1;
        end
    end

    task send_beat(input [127:0] d, input last);
    begin
        s_axis_tdata <= d; s_axis_tvalid <= 1'b1; s_axis_tlast <= last;
        @(posedge clk);
        while (!s_axis_tready) @(posedge clk);
        s_axis_tvalid <= 1'b0; s_axis_tlast <= 1'b0;
    end
    endtask

    // ---------------------------------------------------------------- real helpers
    function real p2(input integer e);
        real r; integer t;
        begin
            r = 1.0;
            if (e >= 0) begin for (t = 0; t < e; t = t + 1) r = r * 2.0; end
            else        begin for (t = 0; t < -e; t = t + 1) r = r * 0.5; end
            p2 = r;
        end
    endfunction

    // real -> float32 bit pattern (truncating; error 2^-23, far below the RTL's 2^-16)
    function [31:0] f32_of(input real v);
        real mm; integer ee; reg [22:0] fr;
        begin
            if (v <= 0.0) f32_of = 32'd0;
            else begin
                ee = 0; mm = v;
                while (mm >= 2.0) begin mm = mm * 0.5; ee = ee + 1; end
                while (mm <  1.0) begin mm = mm * 2.0; ee = ee - 1; end
                fr = (mm - 1.0) * 8388608.0;
                if (fr >= 23'd8388608) fr = 23'd8388607;
                f32_of = {1'b0, (ee + 127), fr};
            end
        end
    endfunction

    // float32 bit pattern -> real, using the precomputed exponent table
    function real f32_to_real(input [31:0] bb);
        begin
            f32_to_real = (1.0 + (bb[22:0] / 8388608.0)) * p2e[bb[30:23]];
        end
    endfunction

    // ---------------------------------------------------------------- golden model
    // Exact integer block dot product, recomputed from the raw int8 / int16 data.
    function integer blk_dot(input integer m_idx, input integer bi);
        integer s, ji;
        begin
            s = 0;
            for (ji = 0; ji < BLOCK; ji = ji + 1)
                s = s + wq[m_idx*k_curr + bi*BLOCK + ji] * xq[bi*BLOCK + ji];
            blk_dot = s;
        end
    endfunction

    // Exactly pl_apply(): accumulate in double, per-block scale = 2^(e-134) * xscale.
    function real calc_row_ref(input integer m_idx, input integer nb);
        real acc;
        integer bi;
        begin
            acc = 0.0;
            for (bi = 0; bi < nb; bi = bi + 1)
                acc = acc + (blk_dot(m_idx, bi) * 1.0) * expsc[we[m_idx*nb + bi]] * xs_dec[bi];
            calc_row_ref = acc;
        end
    endfunction

    // ---------------------------------------------------------------- one test case
    // e_lo/e_hi   : exponent byte range for the random draw
    // xs_octaves  : per-block xscale spread, in octaves (0 = identity xscale path)
    // auto_scale  : 1 => pick the amplitude so |y_ref| ~ 1.2 (realistic regime)
    //               0 => use xscale = 1/16384 as-is
    // exp_sat     : 1 => the row output must sit on an int32 rail
    task run_case(input integer k, input integer mm, input integer sd,
                  input integer dequant, input integer e_lo, input integer e_hi,
                  input integer xs_octaves, input integer auto_scale, input integer exp_sat,
                  input [255:0] label);
        integer nb, xbeats, exp_beats, nout, row_beats, xs_beats, xsb;
        integer nbad;
        real yref, ygot, ymax, ea, er, tol, ypk, yunit, a0, rel, a0f;
        real ycase_max;
        reg signed [31:0] s;
        begin
            ycase_max = 0.0;
            k_curr    = k;
            nb        = k / BLOCK;
            xbeats    = k / 8;
            row_beats = k / 16;
            xs_beats  = (dequant && (xs_octaves != 0)) ? ((nb + 3) / 4) : 0;
            exp_beats = (nb + 15) / 16;
            nout      = dequant ? mm : (mm * nb);

            rstn <= 1'b0;
            repeat (4) @(posedge clk);
            rstn <= 1'b1;
            repeat (3) @(posedge clk);

            seed = sd;
            // x as int16 in [-16384, 16383] (the pl_quant range, max|xq| = 16384)
            for (i = 0; i < k; i = i + 1) begin
                seed = seed * 1103515245 + 12345;
                xq[i] = $signed(seed[15:0]) >>> 1;
            end
            // weights as int8
            for (i = 0; i < mm*k; i = i + 1) begin
                seed = seed * 1103515245 + 12345;
                wq[i] = $signed(seed[15:8]);
            end
            // exponents
            for (i = 0; i < mm*nb; i = i + 1) begin
                seed = seed * 1103515245 + 12345;
                we[i] = e_lo + (seed[7:0] % (e_hi - e_lo + 1));
            end

            // ---- per-block relative amplitude, 2^0 .. 2^xs_octaves
            for (b = 0; b < NB_MAX; b = b + 1) begin
                seed = seed * 1103515245 + 12345;
                rel = 1.0;
                for (i = 0; i < xs_octaves; i = i + 1)
                    if (seed[i] == 1'b1) rel = rel * 2.0;
                xs_rel[b] = rel;
            end

            // ---- pick the amplitude.  y is linear in xscale, so evaluate the
            //      nominal y (xscale = rel/1) and scale it to land near +-1.2.
            if (auto_scale) begin
                ypk = 0.0;
                for (m = 0; m < mm; m = m + 1) begin
                    yunit = 0.0;
                    for (b = 0; b < nb; b = b + 1)
                        yunit = yunit + (blk_dot(m, b) * 1.0)
                                        * expsc[we[m*nb + b]] * xs_rel[b];
                    if (yunit < 0.0) yunit = -yunit;
                    if (yunit > ypk) ypk = yunit;
                end
                a0 = (ypk > 0.0) ? (1.2 * 16384.0 / ypk) : 16384.0;
            end else begin
                a0 = 16384.0;
            end

            // ---- per-block x scale on the wire: xscale[b] = amax_b / 16384
            for (b = 0; b < nb; b = b + 1)
                xsf[b] = (xs_octaves != 0) ? f32_of(a0 * xs_rel[b] / 16384.0) : 32'h3F80_0000;
            for (b = nb; b < NB_MAX; b = b + 1) xsf[b] = 32'h3F80_0000;
            for (b = 0; b < NB_MAX; b = b + 1) xs_dec[b] = f32_to_real(xsf[b]);

            nrecv = 0;
            nlast = 0;

            // ---- header (fields [15:0],[23:16],[30:24],[31] unchanged, [39:32] new)
            beat = 128'd0;
            beat[15:0]  = xbeats;
            beat[23:16] = (nb * 4) - 1;
            beat[30:24] = exp_beats - 1;
            beat[31]    = dequant[0];
            beat[39:32] = xs_beats;
            send_beat(beat, 1'b0);

            // ---- x phase
            for (i = 0; i < xbeats; i = i + 1) begin
                beat = 128'd0;
                for (j = 0; j < 8; j = j + 1)
                    beat[j*16 +: 16] = xq[i*8 + j];
                send_beat(beat, 1'b0);
            end

            // ---- per-block x scale phase (4 float32 per beat, once per token)
            for (i = 0; i < xs_beats; i = i + 1) begin
                beat = 128'd0;
                for (j = 0; j < 4; j = j + 1) begin
                    xsb = i*4 + j;
                    if (xsb < nb) beat[j*32 +: 32] = xsf[xsb];
                end
                send_beat(beat, 1'b0);
            end

            // ---- per row: exponent beats then weight beats
            for (m = 0; m < mm; m = m + 1) begin
                if (dequant) begin
                    for (i = 0; i < exp_beats; i = i + 1) begin
                        beat = 128'd0;
                        for (j = 0; j < 16; j = j + 1)
                            if ((i * 16 + j) < nb) beat[j*8 +: 8] = we[m*nb + i*16 + j];
                        send_beat(beat, 1'b0);
                    end
                end
                for (i = 0; i < row_beats; i = i + 1) begin
                    beat = 128'd0;
                    for (j = 0; j < 16; j = j + 1)
                        beat[j*8 +: 8] = wq[m*k + i*16 + j];
                    send_beat(beat, (m == mm - 1 && i == row_beats - 1));
                end
            end

            // 11 pipeline stages + FIFO drain
            repeat (60) @(posedge clk);

            nbad = 0;
            if (nrecv != nout) begin
                $display("  [FAIL] %0s: expected %0d outputs, got %0d", label, nout, nrecv);
                errors = errors + 1;
            end else if (!dequant) begin
                // legacy mode: exact int32 block dots, bit-exact
                for (m = 0; m < mm; m = m + 1)
                    for (b = 0; b < nb; b = b + 1) begin
                        s = blk_dot(m, b);
                        if (got_out[m*nb + b] !== s) begin
                            if (nbad < 4)
                                $display("    MISMATCH idx %0d: got %0d ref %0d",
                                         m*nb + b, got_out[m*nb + b], s);
                            nbad = nbad + 1;
                        end
                    end
                if (nbad == 0)
                    $display("  [PASS] %0s: %0d raw block dots bit-exact", label, nout);
                else begin
                    $display("  [FAIL] %0s: %0d/%0d raw block dots differ", label, nbad, nout);
                    errors = errors + 1;
                end
            end else begin
                // dequant mode: compare the fixed-point word against the float model
                for (m = 0; m < mm; m = m + 1) begin
                    yref = calc_row_ref(m, nb);
                    ygot = got_out[m] / 1048576.0;      // 2^20
                    ymax = (yref < 0.0) ? -yref : yref;
                    if (ymax > y_abs_max) y_abs_max = ymax;
                    if (ymax > ycase_max) ycase_max = ymax;
                    if (exp_sat != 0) begin
                        // must sit on one of the int32 rails, never wrap or drop
                        if (got_out[m] !== 32'sh7FFF_FFFF && got_out[m] !== 32'sh8000_0000) begin
                            $display("    SAT MISMATCH idx %0d: got %0d, want a rail", m, got_out[m]);
                            nbad = nbad + 1;
                        end
                        if (dbg[9] !== 1'b1) begin
                            $display("    SAT FLAG MISSING idx %0d: sat_sticky not set", m);
                            nbad = nbad + 1;
                        end
                    end else begin
                        ea  = ygot - yref;
                        if (ea < 0.0) ea = -ea;
                        er  = (ymax > 1.0e-6) ? (ea / ymax) : 0.0;
                        tol = 2.0e-6 + 3.0e-4 * ymax;
                        if (ea > err_max_abs) err_max_abs = ea;
                        if (er > err_max_rel) err_max_rel = er;
                        err_ncmp = err_ncmp + 1;
                        if (ea > tol) begin
                            if (nbad < 4)
                                $display("    MISMATCH idx %0d: got %.9f ref %.9f (abs %.3e)",
                                         m, ygot, yref, ea);
                            nbad = nbad + 1;
                        end
                    end
                end
                if (nbad == 0)
                    $display("  [PASS] %0s: %0d rows within tolerance (|y|max %.4f)",
                             label, nout, ycase_max);
                else begin
                    $display("  [FAIL] %0s: %0d/%0d rows out of tolerance", label, nbad, nout);
                    errors = errors + 1;
                end
            end

            // ---- sticky error bits must be clear on every legal burst
            if (exp_sat == 0) begin
                if (dbg_err_tlast || dbg_err_rowbnd || dbg_err_arith) begin
                    $display("  [FAIL] %0s: sticky dbg errors set (tlast=%0b rowbnd=%0b arith=%0b)",
                             label, dbg_err_tlast, dbg_err_rowbnd, dbg_err_arith);
                    errors = errors + 1;
                end
            end
            // ---- exactly one tlast, on the final received beat
            if (nlast != 1) begin
                $display("  [FAIL] %0s: %0d tlast beats (expected 1)", label, nlast);
                errors = errors + 1;
            end
        end
    endtask

    // ---------------------------------------------------------------- main
    initial begin
        for (i = 0; i < 256; i = i + 1) begin
            expsc[i] = p2(i - 134);
            p2e[i]   = p2(i - 127);
        end
    end

    initial begin
        #4000000;
        $display("RESULT: FAIL (timeout at %0t ns)", $time);
        $finish;
    end

    initial begin
        $display("=== tb_gemv_core: independent golden-model regression ===");

        // ---- legacy mode, raw int32 block dots, must be bit-exact
        run_case(128,  3, 32'h1234_5678, 0, 118, 130, 0, 0, 0, "legacy K=128 M=3");
        run_case(1024, 8, 32'h0BAD_F00D, 0, 118, 130, 0, 0, 0, "legacy K=1024 M=8");

        // ---- realistic dequant mode: e in 118..130, 16x xscale spread across blocks
        run_case(128,  3, 32'hA5A5_5A5A, 1, 118, 130, 4, 1, 0, "dequant K=128 M=3");
        run_case(1024, 8, 32'hCAFE_BABE, 1, 118, 130, 4, 1, 0, "dequant K=1024 M=8");
        run_case(128,  8, 32'h9876_5432, 1, 118, 130, 4, 1, 0, "dequant K=128 M=8");
        // wide per-block xscale spread (1024x across the blocks of one row)
        run_case(1024, 4, 32'h5EED_1234, 1, 118, 130, 10, 1, 0, "dequant K=1024 M=4 wide-xs");
        // flat xscale
        run_case(512,  4, 32'h2468_ACE0, 1, 118, 130, 1, 1, 0, "dequant K=512 M=4 flat-xs");
        // identity xscale path (xs_beats = 0); exponents chosen so |y| stays in range
        run_case(512,  4, 32'h1357_9BDF, 1, 104, 116, 0, 0, 0, "dequant K=512 M=4 no-xs");

        // ---- saturation: huge exponent must clamp to a rail, not wrap and not drop.
        //      nb = 1 so one block decides the row and the outcome is deterministic.
        run_case(64,   2, 32'hDEAD_BEEF, 1, 240, 250, 0, 0, 1, "saturate to int32 rail");
        // ---- flush: tiny exponent must give ~0 (not a bogus value)
        run_case(64,   2, 32'hF00D_4321, 1,  40,  60, 0, 0, 0, "flush K=64 M=2");

        $display("");
        $display("--- numeric error over all non-saturating dequant cases ---");
        $display("  compared rows      : %0d", err_ncmp);
        $display("  max |y_ref|        : %.6e", y_abs_max);
        $display("  max abs error      : %.6e", err_max_abs);
        $display("  max relative error : %.6e", err_max_rel);
        $display("");
        if (errors == 0) $display("RESULT: ALL PASS");
        else             $display("RESULT: FAIL (%0d errors)", errors);
        $finish;
    end

endmodule
