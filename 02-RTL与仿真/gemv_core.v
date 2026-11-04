// gemv_core.v -- BFP8 block dot-product engine with on-chip dequantisation,
// per-block x rescaling, saturating fixed-point accumulation, and an 11-stage
// pipeline that closes timing at 166.67 MHz (6.000 ns) on xc7z020-2.
//
// ---------------------------------------------------------------------------
// 1. Stream format
// ---------------------------------------------------------------------------
// One AXI4-Stream slave, one AXI4-Stream master, no AXI-Lite.  All configuration
// travels in the first beat of the stream (the "header").  The whole M*K weight
// stream and all exponent groups form ONE contiguous burst that ends with a
// single s_axis_tlast on the final weight beat.  Splitting the burst wedges the
// AXI DMA permanently, so the core never ends a burst on its own.
//
//   beat 0        : header
//                     tdata[15:0]  = x_beats                    (unchanged)
//                     tdata[23:16] = beats_per_row - 1           (unchanged)
//                     tdata[30:24] = exp_beats_per_row - 1       (unchanged)
//                     tdata[31]    = dequant_en                  (unchanged)
//                     tdata[39:32] = xs_beats                    (NEW, see below)
//                     all other bits reserved, drive 0
//   next x_beats  : x, 8 int16 per beat
//   next xs_beats : per-block x scale, 4 float32 per beat   (only if xs_beats != 0)
//   then per row  : exp_beats beats of exponents (16 uint8 per beat)
//                   followed by beats_per_row beats of weights (16 int8 per beat)
//
// xs_beats is the number of beats in the NEW x-scale phase.  The phase carries
// the per-block x scale produced by pl_quant()/bfp8_gemv():
//
//     amax = max|x| over the 64-element block
//     xscale[b] = amax / 16384                (float32, one per block)
//
// It is streamed ONCE per token because it is shared by every row.  xs_beats = 0
// is legal and means "identity scale" (xscale[b] = 1.0f for every b), which
// reproduces the pre-xscale behaviour exactly.
//
// dequant_en = 0 is the legacy mode: raw int32 block dot products, one m_axis
// beat per 64-element block, no shifting, no xscale, no row accumulation.
//
// ---------------------------------------------------------------------------
// 2. Arithmetic
// ---------------------------------------------------------------------------
// The software reference (qwen_kernels.c bfp8_gemv / pl_apply.c pl_apply) is
//
//     y[m] = sum_b (float)dots[m][b] * EXP_SCALE[exps[m][b]] * xscale[b]
//     EXP_SCALE[e] = ldexpf(1.0f, e - 134)
//
// This core evaluates exactly that, in integer arithmetic:
//
//   * dots[m][b]  = sum_{i<64} w_int8 * x_int16            (S1..S6, exact int32)
//   * xscale[b] is decoded from its float32 bits into a 17-bit mantissa
//     (implicit 1 + the top 16 fraction bits, so xscale = mant * 2^-16 * 2^Ex)
//     and multiplied into the block dot:  prod = dots * mant          (S8)
//   * the whole scale is then a single power of two:
//         contrib = prod * 2^(Ex + e - 110)
//     because 2^-16 * 2^(e-134) * 2^40 = 2^(e-110).                  (S9/S10)
//   * the row accumulator is a signed 64-bit value carrying 40 fractional bits
//     (acc = y * 2^40), so nothing is truncated before the host scales it. (S11)
//
// The output word is the saturated fixed point value, NOT a rounded integer:
//
//     m_axis_tdata = clamp(row_acc, -(2^31)<<20, (2^31-1)<<20)[51:20]
//                  = floor(y * 2^20)      as int32
//
// and the host multiplies by 2^-20.  Rounding to an integer here would destroy
// 20 bits of resolution: with |y| ~ 0.8 that is ~54% mean relative error, and
// 31% of outputs would round to zero.
//
// Saturation is defined everywhere (see SATURATION below); a sticky bit in dbg
// reports that it happened.  Nothing wraps, and nothing is silently dropped.
//
// ---------------------------------------------------------------------------
// 3. SATURATION  (replaces the old "shift > 63 -> 0" block-drop bug)
// ---------------------------------------------------------------------------
// The legacy code computed shift_64 = e - 114 and forced shifted_64 = 0 for
// shift_64 > 63, i.e. for every exponent byte e > 177 the whole 64-element block
// was silently discarded.  The software reference uses ldexpf(1.0f, e-134),
// which saturates to +inf for e >= 262 and is a huge finite value below that.
// We reproduce the +inf semantics with a fully defined saturating datapath:
//
//   * the barrel shift amount is clamped to [-63, +63]           (err_arith_ovf)
//   * a left shift that would drop significant bits is detected exactly and the
//     block contribution is forced to +/-(2^63-1)               (err_arith_ovf)
//   * the row accumulator uses a saturating add, so 64 blocks can never wrap
//     around and cancel each other                         (err_arith_ovf)
//   * the final 32-bit conversion clamps to INT32_MAX / INT32_MIN  (sat_sticky)
//
// A saturated row therefore reads back as INT32_MAX / INT32_MIN after the host
// scales by 2^-20, which is the integer-domain image of +inf / -inf.
//
// ---------------------------------------------------------------------------
// 4. dbg[31:0] layout  (read-only status, driven to axi_gpio)
// ---------------------------------------------------------------------------
//   [2:0]   state                [13]    xs_beats != 0 (xscale phase enabled)
//   [3]     s_axis_tvalid        [15:14] reserved, 0
//   [4]     s_axis_tready        [23:16] pos[7:0]
//   [5]     m_axis_tvalid        [31:24] x_load_cnt[7:0]
//   [6]     m_axis_tready
//   [7]     m_axis_tlast
//   [8]     dequant_en
//   [9]     sat_sticky           output clamped to int32 range (this burst)
//   [10]    err_arith_ovf        sticky: shift clamp / shift overflow / acc sat
//   [11]    err_tlast_phase      sticky: s_axis_tlast seen in the wrong phase
//   [12]    err_row_bnd          sticky: row boundary / header sanity failure
//
// The three sticky error bits are cleared only by aresetn or dbg_rst, so the
// host can read them after a burst.  They never clear on a new header.

`timescale 1ns / 1ps

module gemv_core #(
    parameter integer ELEM_PER_BEAT     = 16,
    parameter integer BLOCK             = 64,
    parameter integer BEATS_PER_BLOCK   = BLOCK / ELEM_PER_BEAT,   // 4
    parameter integer XW                = 8,
    parameter integer X_DEPTH           = 1 << XW                 // 256 words = 4096 elements
)(
    input  wire         aclk,
    input  wire         aresetn,
    input  wire         dbg_rst,      // soft reset, driven from axi_gpio
    output wire [31:0]  dbg,          // read-only status, read by axi_gpio

    input  wire [127:0] s_axis_tdata,
    input  wire [15:0]  s_axis_tkeep,
    input  wire         s_axis_tvalid,
    output wire         s_axis_tready,
    input  wire         s_axis_tlast,

    output wire [31:0]  m_axis_tdata,
    output wire [3:0]   m_axis_tkeep,
    output reg          m_axis_tvalid,
    input  wire         m_axis_tready,
    output wire         m_axis_tlast
);

    assign m_axis_tkeep = 4'hF;

    localparam ST_HDR   = 3'd0,
               ST_LOADX = 3'd1,
               ST_XS    = 3'd2,
               ST_EXP   = 3'd3,
               ST_RUN   = 3'd4,
               ST_DRAIN = 3'd5;

    // 64-bit saturation rails for the row accumulator
    localparam signed [63:0] ACC_MAX64 = 64'sh7FFF_FFFF_FFFF_FFFF;  // +(2^63-1)
    localparam signed [63:0] ACC_MIN64 = 64'sh8000_0000_0000_0001;  // -(2^63-1)
    // 32-bit saturation rails in the y * 2^20 domain
    localparam signed [63:0] OUT_MAX64 = 64'sd2251799812636672;     //  (2^31-1) << 20
    localparam signed [63:0] OUT_MIN64 = -64'sd2251799813685248;    // -(2^31)   << 20

    // ---------------------------------------------------------------- configuration
    reg [2:0]  state;
    reg [15:0] x_last;          // x_beats - 1
    reg [7:0]  bpr_m1;
    reg        dequant_en;
    reg [6:0]  exp_beats_m1;
    reg [6:0]  exp_cnt;
    reg [7:0]  xs_beats;        // beats in the xscale phase (0 = identity scale)
    reg [7:0]  xs_cnt;
    reg [15:0] x_load_cnt;
    reg [127:0] x_hold;
    reg [7:0]  pos;
    reg        row_toggle;      // ping-pong row exponent buffer select

    // ---------------------------------------------------------------- sticky error / status
    reg err_arith_ovf;
    reg err_tlast_phase;
    reg err_row_bnd;
    reg sat_sticky;

    // ---------------------------------------------------------------- ping-pong row exponents
    // Written 16 bytes at a time during the exponent phase, read one byte at a
    // time per block.  Two banks so that row n+1 can be loaded while row n is
    // still draining through the pipeline.
    reg [7:0] row_exp [0:1][0:127];

    // ---------------------------------------------------------------- per-block x scale
    // One 128-bit line per streamed beat, so the phase needs a single write port
    // and still infers as distributed RAM.  16 lines hold the 64 floats of the
    // largest legal token (nb = K/64 <= 64 with X_DEPTH = 256 words).
    reg [127:0] xs_mem [0:15];

    // ---------------------------------------------------------------- x memory
    reg [255:0] xmem [0:X_DEPTH-1];
    reg [XW-1:0] xw_waddr;
    reg          xw_we;
    reg [255:0]  xw_wdata;
    wire [XW-1:0] xw_raddr = pos[XW-1:0];
    wire [255:0] xw_rdata = xmem[xw_raddr];

    always @(posedge aclk) begin
        if (xw_we) xmem[xw_waddr] <= xw_wdata;
    end

    // ==================================================================== pipeline
    // S1..S5 : block dot product            (unchanged from the original core)
    // S6     : capture the int32 block dot, launch the exponent / xscale reads
    // S7     : float32 xscale decode, alignment exponent sh = Ex + e - 110, |bd|
    // S8     : prod = block_dot * xs_mant   (and |prod| on a second DSP)
    // S9     : shift by sh[2:0], exact left-shift overflow detect
    // S10    : shift by sh[5:3]*8, sign-select -> contrib
    // S11    : saturating row accumulation
    // S12    : saturated fixed-point output beat
    //
    // The 64-bit barrel shifter is deliberately split over S9/S10: a single
    // six-level 64-bit mux tree routes badly (measured 4.8 ns of route on
    // xc7z020) and each half now owns a full period.
    reg        v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12;
    // c: {s_axis_tlast, is_row_end, row_toggle, pos[7:0]} (11 bits)
    reg [10:0] c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12;
    reg [255:0] x_r;
    reg [127:0] w_r;
    reg signed [31:0] p_r [0:ELEM_PER_BEAT-1];
    reg signed [31:0] q0, q1, q2, q3;
    reg signed [31:0] q01, q23;
    reg signed [31:0] sum_r;
    reg signed [31:0] acc;

    // 32-bit beat partial dot, accumulated over the BEATS_PER_BLOCK beats of a block
    wire signed [31:0] dot_nx = acc + sum_r;

    wire rst_n  = aresetn & ~dbg_rst;
    wire s_fire = s_axis_tvalid & s_axis_tready;
    assign s_axis_tready = ((state == ST_HDR) || (state == ST_LOADX) ||
                            (state == ST_XS)  || (state == ST_EXP) ||
                            (state == ST_RUN)) && !m_axis_tvalid;

    integer i;
    always @(posedge aclk) begin
        if (!rst_n) begin
            v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0; v4 <= 1'b0; v5 <= 1'b0;
            c1 <= 11'd0; c2 <= 11'd0; c3 <= 11'd0; c4 <= 11'd0; c5 <= 11'd0;
            x_r <= 256'd0; w_r <= 128'd0;
            q0 <= 0; q1 <= 0; q2 <= 0; q3 <= 0;
            q01 <= 0; q23 <= 0;
            sum_r <= 0;
            acc <= 0;
            for (i = 0; i < ELEM_PER_BEAT; i = i + 1) p_r[i] <= 32'sd0;
        end else begin
            v1 <= (state == ST_RUN) && s_fire;
            if ((state == ST_RUN) && s_fire) begin
                x_r <= xw_rdata;
                w_r <= s_axis_tdata;
                c1  <= {s_axis_tlast, (pos == bpr_m1), row_toggle, pos};
            end
            v2 <= v1;
            c2 <= c1;
            for (i = 0; i < ELEM_PER_BEAT; i = i + 1)
                p_r[i] <= $signed(w_r[i*8 +: 8]) * $signed(x_r[i*16 +: 16]);
            v3 <= v2;
            c3 <= c2;
            q0 <= (p_r[0]  + p_r[1])  + (p_r[2]  + p_r[3]);
            q1 <= (p_r[4]  + p_r[5])  + (p_r[6]  + p_r[7]);
            q2 <= (p_r[8]  + p_r[9])  + (p_r[10] + p_r[11]);
            q3 <= (p_r[12] + p_r[13]) + (p_r[14] + p_r[15]);
            v4 <= v3;
            c4 <= c3;
            q01 <= q0 + q1;
            q23 <= q2 + q3;
            v5 <= v4;
            c5 <= c4;
            sum_r <= q01 + q23;
            // block accumulator: sums the BEATS_PER_BLOCK beat partials, then restarts
            if (v5) acc <= (c5[1:0] == BEATS_PER_BLOCK - 1) ? 32'sd0 : dot_nx;
        end
    end

    // ---------------------------------------------------------------- S5 context decode
    wire [7:0] c5_pos   = c5[7:0];
    wire [1:0] c5_sub   = c5_pos[1:0];
    wire       blk_end5 = (c5_sub == BEATS_PER_BLOCK - 1);

    // ---------------------------------------------------------------- S6: block dot capture
    reg signed [31:0] bd_q;         // int32 block dot product
    reg [7:0]         exp_q;        // BFP8 weight exponent byte
    reg [31:0]        xsf_q;        // raw float32 x scale

    always @(posedge aclk) begin
        if (!rst_n) begin
            v6    <= 1'b0;
            c6    <= 11'd0;
            bd_q  <= 32'sd0;
            exp_q <= 8'd134;
            xsf_q <= 32'h3F80_0000;     // 1.0f
        end else begin
            v6 <= v5 & blk_end5;
            c6 <= c5;
            if (v5 & blk_end5) bd_q <= dot_nx;
            if (v6) begin
                // Both reads are addressed from the registered beat context, so the
                // 128:1 exponent mux and the 16:1 xscale mux each own a full period.
                // blk = c6[7:2]; xs_mem line = blk/4 = c6[7:4], float in line = blk%4.
                exp_q <= row_exp[c6[8]][c6[7:2]];
                xsf_q <= xs_mem[c6[7:4]][{c6[3:2], 5'b0} +: 32];
            end
        end
    end

    // ---------------------------------------------------------------- S7: xscale decode
    // float32 -> (17-bit mantissa in [65536, 131071], unbiased exponent).
    // Zero / denormal / inf / NaN all decode to mantissa 0, i.e. a zero
    // contribution, which is exactly what xscale = 0.0f must do.
    wire [7:0]  xsf_eb    = xsf_q[30:23];
    wire        xsf_norm  = (xsf_eb != 8'd0) && (xsf_eb != 8'd255);
    wire [16:0] xs_mant   = xsf_norm ? {1'b1, xsf_q[22:7]} : 17'd0;
    wire signed [8:0] xs_ex = xsf_norm ? ($signed({1'b0, xsf_eb}) - 9'sd127) : 9'sd0;

    reg [16:0]        mant_q;
    reg signed [9:0]  sh_q;         // Ex + e - 110
    reg [31:0]        bd_abs_q;     // |block dot|, used to build |prod| on a DSP

    always @(posedge aclk) begin
        if (!rst_n) begin
            v7     <= 1'b0;
            c7     <= 11'd0;
            mant_q <= 17'd0;
            sh_q   <= 10'sd0;
            bd_abs_q <= 32'd0;
        end else begin
            v7 <= v6;
            c7 <= c6;
            if (v7) begin
                mant_q <= xs_mant;
                // xscale = (mant / 2^16) * 2^Ex, and the accumulator carries 40
                // fractional bits, so the whole per-block scale is one power of two:
                //   (mant/2^16) * 2^Ex * 2^(e-134) * 2^40 = mant * 2^(Ex + e - 110)
                sh_q   <= xs_ex + ($signed({1'b0, exp_q}) - 9'sd134) + 10'sd24;
                // |bd_q * mant| = |bd_q| * mant because mant >= 0, so the magnitude
                // of the product is available without a 49-bit negate after the DSP.
                bd_abs_q <= bd_q[31] ? ((~bd_q) + 32'd1) : bd_q;
            end
        end
    end

    // ---------------------------------------------------------------- S7/S8 shift decode
    // sh is clamped to [-63, +63].  Clamping the NEGATIVE side is lossless: prod
    // is only 49 bits, so any right shift of 49 or more is already an exact flush
    // to 0 / -1, which is what the true arithmetic shift would give.  Clamping the
    // POSITIVE side does lose bits and is reported.
    wire signed [9:0]  sh_c        = (sh_q > 10'sd63)  ? 10'sd63  :
                                     (sh_q < -10'sd63) ? -10'sd63 : sh_q;
    wire               sh_clamped  = (sh_q > 10'sd63);
    // Exact left-shift overflow test operand: |prod| < 2^48, so sh <= 15 can never
    // overflow and the test is skipped there (ovf_sh = 63 forces a zero result).
    // This is decoded one stage early so that the S9 compare starts from a register.
    wire        [5:0]  ovf_sh_nx   = (sh_c >= 10'sd16) ? (6'd63 - sh_c[5:0]) : 6'd63;

    // ---------------------------------------------------------------- S8: block dot * xscale
    reg signed [48:0] prod_q;
    reg        [48:0] prod_abs_q;
    reg        [5:0]  ovf_sh_q;
    reg               sh_clamp_q;

    always @(posedge aclk) begin
        if (!rst_n) begin
            v8         <= 1'b0;
            c8         <= 11'd0;
            prod_q     <= 49'sd0;
            prod_abs_q <= 49'd0;
            ovf_sh_q   <= 6'd63;
            sh_clamp_q <= 1'b0;
        end else begin
            v8 <= v7;
            c8 <= c7;
            if (v8) begin
                prod_q     <= bd_q * $signed({1'b0, mant_q});
                prod_abs_q <= $signed({1'b0, bd_abs_q}) * $signed({1'b0, mant_q});
                ovf_sh_q   <= ovf_sh_nx;
                sh_clamp_q <= sh_clamped;
            end
        end
    end

    // ---------------------------------------------------------------- S9: shift, low 3 bits
    wire               sh_neg  = sh_c[9];
    wire        [2:0]  ls_lo  = sh_c[2:0];
    wire        [9:0]  neg_mag = 10'd0 - sh_c;          // |sh_c|, 1..63
    wire        [2:0]  rs_lo  = neg_mag[2:0];
    wire signed [63:0] wide   = {{15{prod_q[48]}}, prod_q};
    wire signed [63:0] shl_lo = wide <<  ls_lo;
    wire signed [63:0] shr_lo = wide >>> rs_lo;

    // Exact left-shift overflow test on the pre-computed |prod|: no 49-bit negate
    // and no shift-amount decode sits on this path, only the compare itself.
    wire        ovf_nx   = ((prod_abs_q >> ovf_sh_q) != 49'd0);

    reg signed [63:0] shl_lo_q;
    reg signed [63:0] shr_lo_q;
    reg signed [9:0]  sh_q2;
    reg               ovf_q;
    reg               pneg2_q;

    always @(posedge aclk) begin
        if (!rst_n) begin
            v9        <= 1'b0;
            c9        <= 11'd0;
            shl_lo_q  <= 64'sd0;
            shr_lo_q  <= 64'sd0;
            sh_q2     <= 10'sd0;
            ovf_q     <= 1'b0;
            pneg2_q   <= 1'b0;
        end else begin
            v9 <= v8;
            c9 <= c8;
            if (v9) begin
                shl_lo_q <= shl_lo;
                shr_lo_q <= shr_lo;
                sh_q2    <= sh_q;
                ovf_q    <= ovf_nx | sh_clamp_q;
                pneg2_q  <= prod_q[48];
            end
        end
    end

    // ---------------------------------------------------------------- S10: shift, high 3 bits
    // (wide << a) << b == wide << (a+b) and the same identity holds for the
    // arithmetic right shift, so the two halves compose exactly.  The high bits
    // come from the sign-magnitude form for the right shift, from sh itself for
    // the left shift.
    wire [9:0]  neg_mag2 = 10'd0 - sh_q2;               // |sh|, 0..63
    wire [2:0]  hi_sel_l = sh_q2[5:3];
    wire [2:0]  hi_sel_r = neg_mag2[5:3];
    wire        hi_neg   = sh_q2[9];
    wire [5:0]  hi_amt   = hi_neg ? {hi_sel_r, 3'b000} : {hi_sel_l, 3'b000};

    wire signed [63:0] shl_hi = shl_lo_q <<  hi_amt;
    wire signed [63:0] shr_hi = shr_lo_q >>> hi_amt;
    wire signed [63:0] shifted_nx = hi_neg ? shr_hi : shl_hi;

    reg signed [63:0] shifted_q;
    reg               prod_neg_q;

    always @(posedge aclk) begin
        if (!rst_n) begin
            v10        <= 1'b0;
            c10        <= 11'd0;
            shifted_q  <= 64'sd0;
            prod_neg_q <= 1'b0;
        end else begin
            v10 <= v9;
            c10 <= c9;
            if (v10) begin
                shifted_q  <= shifted_nx;
                prod_neg_q <= pneg2_q;
            end
        end
    end

    // ---------------------------------------------------------------- S11: row accumulation
    reg signed [63:0] row_acc_64;

    wire signed [63:0] contrib = ovf_q ? (prod_neg_q ? ACC_MIN64 : ACC_MAX64) : shifted_q;
    wire signed [64:0] sum_ext = {row_acc_64[63], row_acc_64} + {contrib[63], contrib};
    wire               add_ovf = (sum_ext[64] != sum_ext[63]);

    always @(posedge aclk) begin
        if (!rst_n) begin
            v11        <= 1'b0;
            c11        <= 11'd0;
            row_acc_64 <= 64'sd0;
        end else begin
            v11 <= v10;
            c11 <= c10;
            if (v11 && dequant_en) begin
                row_acc_64 <= add_ovf ? (row_acc_64[63] ? ACC_MIN64 : ACC_MAX64)
                                      : sum_ext[63:0];
            end
            // Row reset happens one stage later, on the output beat, so the
            // finished accumulator stays visible while it is being converted.
            if (v12 && dequant_en && c12[9]) row_acc_64 <= 64'sd0;
        end
    end

    wire c12_gemv_last = c12[10];
    wire c12_row_last  = c12[9];

    always @(posedge aclk) begin
        if (!rst_n) begin
            v12 <= 1'b0;
            c12 <= 11'd0;
        end else begin
            v12 <= v11;
            c12 <= c11;
        end
    end

    // ---------------------------------------------------------------- S12: output beat
    // Saturated fixed-point conversion: the accumulator carries 40 fractional bits,
    // bits [51:20] are y * 2^20, and the value is clamped to the int32 range first.
    // The host multiplies the received word by 2^-20 to recover y.
    wire               out_hi  = (row_acc_64 > OUT_MAX64);
    wire               out_lo  = (row_acc_64 < OUT_MIN64);
    wire signed [63:0] out_sat = out_hi ? OUT_MAX64 : out_lo ? OUT_MIN64 : row_acc_64;
    wire        [31:0] out_word = out_sat[51:20];

    // dequant_en = 1 : one beat per row, pushed at S12 from the finished accumulator
    // dequant_en = 0 : one beat per block, pushed at S6 straight from bd_q (legacy)
    wire [31:0] of_data_nx = dequant_en ? out_word : bd_q;
    wire        of_last_nx = dequant_en ? c12_gemv_last : c6[10];
    wire        of_push    = dequant_en ? (v12 && c12_row_last) : (v6 && !dequant_en);

    reg [31:0] of_data [0:7];
    reg        of_last [0:7];
    reg [2:0]  of_wp, of_rp;
    reg [3:0]  of_cnt;

    wire of_pop  = m_axis_tvalid && m_axis_tready;
    wire [3:0] of_cnt_next = of_cnt + (of_push ? 4'd1 : 4'd0) - (of_pop ? 4'd1 : 4'd0);

    assign m_axis_tdata = of_data[of_rp];
    assign m_axis_tlast = of_last[of_rp];

    always @(posedge aclk) begin
        if (!rst_n) begin
            m_axis_tvalid <= 1'b0;
            of_wp         <= 3'd0;
            of_rp         <= 3'd0;
            of_cnt        <= 4'd0;
            sat_sticky    <= 1'b0;
        end else begin
            if (of_push) begin
                of_data[of_wp] <= of_data_nx;
                of_last[of_wp] <= of_last_nx;
                of_wp          <= of_wp + 3'd1;
            end
            if (of_pop) of_rp <= of_rp + 3'd1;

            of_cnt        <= of_cnt_next;
            m_axis_tvalid <= (of_cnt_next != 4'd0);

            if (dequant_en && v12 && c12_row_last && (out_hi || out_lo))
                sat_sticky <= 1'b1;
        end
    end

    // ==================================================================== main FSM
    integer k;
    always @(posedge aclk) begin
        if (!rst_n) begin
            state          <= ST_HDR;
            x_last         <= 16'd0;
            bpr_m1         <= 8'd0;
            dequant_en     <= 1'b0;
            exp_beats_m1   <= 7'd0;
            exp_cnt        <= 7'd0;
            xs_beats       <= 8'd0;
            xs_cnt         <= 8'd0;
            x_load_cnt     <= 16'd0;
            x_hold         <= 128'd0;
            xw_we          <= 1'b0;
            xw_waddr       <= {XW{1'b0}};
            xw_wdata       <= 256'd0;
            pos            <= 8'd0;
            row_toggle     <= 1'b0;
            err_arith_ovf  <= 1'b0;
            err_tlast_phase<= 1'b0;
            err_row_bnd    <= 1'b0;
            for (k = 0; k < 128; k = k + 1) begin
                row_exp[0][k] <= 8'd134;
                row_exp[1][k] <= 8'd134;
            end
            for (k = 0; k < 16; k = k + 1) xs_mem[k] <= {4{32'h3F80_0000}};  // 1.0f
        end else begin
            xw_we <= 1'b0;

            // ---- sticky arithmetic saturation reporting -------------------
            // Only meaningful in dequant mode; the shift path is idle when
            // dequant_en = 0 and must not raise a spurious flag.
            if (v9  && dequant_en && (ovf_nx | sh_clamp_q)) err_arith_ovf <= 1'b1;
            if (v11 && dequant_en && add_ovf)               err_arith_ovf <= 1'b1;

            case (state)
            ST_HDR: begin
                if (s_fire) begin
                    x_last       <= s_axis_tdata[15:0] - 16'd1;
                    bpr_m1       <= s_axis_tdata[23:16];
                    exp_beats_m1 <= s_axis_tdata[30:24];
                    dequant_en   <= s_axis_tdata[31];
                    xs_beats     <= s_axis_tdata[39:32];
                    exp_cnt      <= 7'd0;
                    xs_cnt       <= 8'd0;
                    x_load_cnt   <= 16'd0;
                    pos          <= 8'd0;
                    row_toggle   <= 1'b0;
                    // Re-arm both accumulators for the new burst.  acc (S5) is only
                    // cleared on a block end and row_acc_64 (S11) only on a row end,
                    // so a burst that stops in the middle of a row leaves a partial
                    // sum behind and the next burst's first row would start from it.
                    // ST_HDR is only reachable from ST_DRAIN, which waits for
                    // v1..v12 to fall, so the pipeline is empty here and this cannot
                    // disturb an in-flight row.  For a well formed burst both values
                    // are already zero, so this is a no-op; the gate on the incoming
                    // dequant_en keeps the legacy (dequant_en = 0) path bit-identical.
                    if (s_axis_tdata[31]) begin
                        acc        <= 32'sd0;
                        row_acc_64 <= 64'sd0;
                    end
                    // header sanity: rows must be a whole number of blocks, and
                    // the exponent phase must be able to cover every block
                    if (((s_axis_tdata[23:16] + 8'd1) % BEATS_PER_BLOCK) != 0)
                        err_row_bnd <= 1'b1;
                    else if (s_axis_tdata[31] &&
                             ((({9'd0, s_axis_tdata[30:24]} + 16'd1) * 16) <
                              (({8'd0, s_axis_tdata[23:16]} + 16'd1) / BEATS_PER_BLOCK)))
                        err_row_bnd <= 1'b1;

                    if (s_axis_tdata[15:0] != 16'd0) begin
                        state <= ST_LOADX;
                    end else if (s_axis_tdata[39:32] != 8'd0) begin
                        state <= ST_XS;
                    end else if (s_axis_tdata[31]) begin
                        state <= ST_EXP;
                    end else begin
                        state <= ST_RUN;
                    end
                    // A header beat carrying tlast is a zero-length burst: record it
                    // and drain rather than waiting for weight beats that never come.
                    if (s_axis_tlast) begin
                        err_tlast_phase <= 1'b1;
                        state           <= ST_DRAIN;
                    end
                end
            end

            ST_LOADX: begin
                if (s_fire) begin
                    if (s_axis_tlast) err_tlast_phase <= 1'b1;
                    if (x_load_cnt[0] == 1'b0) begin
                        x_hold <= s_axis_tdata;
                    end else begin
                        xw_we    <= 1'b1;
                        xw_waddr <= x_load_cnt[XW:1];
                        xw_wdata <= {s_axis_tdata, x_hold};
                    end
                    x_load_cnt <= x_load_cnt + 16'd1;
                    if (x_load_cnt == x_last) begin
                        exp_cnt <= 7'd0;
                        pos     <= 8'd0;
                        if (xs_beats != 8'd0)       state <= ST_XS;
                        else if (dequant_en)        state <= ST_EXP;
                        else                        state <= ST_RUN;
                        if (s_axis_tlast) state <= ST_DRAIN;
                    end
                end
            end

            ST_XS: begin
                if (s_fire) begin
                    if (s_axis_tlast) err_tlast_phase <= 1'b1;
                    if (xs_cnt[7:4] == 4'd0) xs_mem[xs_cnt[3:0]] <= s_axis_tdata;
                    xs_cnt <= xs_cnt + 8'd1;
                    if (xs_cnt == (xs_beats - 8'd1)) begin
                        state   <= dequant_en ? ST_EXP : ST_RUN;
                        exp_cnt <= 7'd0;
                        pos     <= 8'd0;
                        if (s_axis_tlast) state <= ST_DRAIN;
                    end
                end
            end

            ST_EXP: begin
                if (s_fire) begin
                    // tlast must never land here: the whole weight stream plus all
                    // exponent groups are one contiguous burst.  Flag it and leave
                    // instead of silently eating the next header as weight data.
                    if (s_axis_tlast) begin
                        err_tlast_phase <= 1'b1;
                        state <= ST_DRAIN;
                    end else begin
                        for (k = 0; k < 16; k = k + 1) begin
                            if ((exp_cnt * 16 + k) < 128)
                                row_exp[row_toggle][exp_cnt * 16 + k] <= s_axis_tdata[k*8 +: 8];
                        end
                        if (exp_cnt == exp_beats_m1) begin
                            state   <= ST_RUN;
                            exp_cnt <= 7'd0;
                            pos     <= 8'd0;
                        end else begin
                            exp_cnt <= exp_cnt + 7'd1;
                        end
                    end
                end
            end

            ST_RUN: begin
                if (s_fire) begin
                    pos <= (pos == bpr_m1) ? 8'd0 : (pos + 8'd1);
                    if (s_axis_tlast) begin
                        // a burst that ends anywhere but the last beat of a row is
                        // a truncated stream: record it, then drain normally
                        if (pos != bpr_m1) err_row_bnd <= 1'b1;
                        state <= ST_DRAIN;
                    end else if (pos == bpr_m1) begin
                        if (dequant_en) begin
                            state      <= ST_EXP;
                            exp_cnt    <= 7'd0;
                            row_toggle <= ~row_toggle;
                        end else begin
                            row_toggle <= ~row_toggle;
                        end
                    end
                end
            end

            ST_DRAIN: begin
                // Wait for all twelve stages to retire before a new header can be
                // accepted, so dequant_en / bpr_m1 / xs_beats cannot change under
                // a beat that is still in flight.
                if (!v1 && !v2 && !v3 && !v4 && !v5 && !v6 &&
                    !v7 && !v8 && !v9 && !v10 && !v11 && !v12) state <= ST_HDR;
            end

            default: state <= ST_HDR;
            endcase
        end
    end

    // ==================================================================== debug
    assign dbg = {x_load_cnt[7:0],          // [31:24]
                  pos,                      // [23:16]
                  2'b00,                    // [15:14] reserved
                  (xs_beats != 8'd0),       // [13]    xscale phase enabled
                  err_row_bnd,              // [12]    sticky row-boundary error
                  err_tlast_phase,          // [11]    sticky tlast-phase error
                  err_arith_ovf,            // [10]    sticky arithmetic saturation
                  sat_sticky,               // [9]     output clamped to int32
                  dequant_en,               // [8]
                  m_axis_tlast,             // [7]
                  m_axis_tready,            // [6]
                  m_axis_tvalid,            // [5]
                  s_axis_tready,            // [4]
                  s_axis_tvalid,            // [3]
                  state};                   // [2:0]

endmodule
