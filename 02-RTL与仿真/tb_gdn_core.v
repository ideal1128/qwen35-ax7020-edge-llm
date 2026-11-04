// tb_gdn_core.v -- self-checking testbench for gdn_core
//
// Reference model (integer, mirrors the RTL arithmetic exactly):
//   S[i][j] : Q4.12 signed 16, initialised to 0 by mode=1
//   u[j]    = sum_i k[i]*S[i][j]                      (Q6.26, 32-bit)
//   u_q[j]  = sat16(u[j], 12)                          (Q2.14)
//   e[j]    = sat16(v[j] - u_q[j], 0)
//   S'[i][j]= sat16( (alpha*S[i][j])>>>16 + (beta*k[i]*e[j])>>>31 , 0 )
//   o[j]    = sum_i q[i]*S'[i][j]                      (Q6.26)
//   o_q[j]  = sat16(o[j], 12)                          (Q2.14)
//
// Checks:
//   A. mode=1 (head_idx=8, all heads) then mode=2 dump of head 0 -> 4096 zeros
//   B. one mode=0 token -> 64 output beats match the reference
//   C. mode=2 dump after the token -> 4096 state words match the reference
//   D. the dump is also compared with the documented "e is Q3.13" model
//      (beta term >>>32) so a format mismatch is reported with numbers
//   E. mode=1 with head_idx=0 clears head 0 only
`timescale 1ns / 1ps

module tb_gdn_core;

    reg         aclk;
    reg         aresetn;

    reg  [63:0] s_tdata;
    reg         s_tvalid;
    wire        s_tready;
    reg         s_tlast;

    wire [31:0] m_tdata;
    wire        m_tvalid;
    reg         m_tready;
    wire        m_tlast;

    reg  [7:0]  mode;
    reg  [7:0]  head_idx;
    reg  [15:0] alpha;
    reg  [15:0] beta;

    gdn_core dut (
        .aclk         (aclk),
        .aresetn      (aresetn),
        .s_axis_tdata (s_tdata),
        .s_axis_tvalid(s_tvalid),
        .s_axis_tready(s_tready),
        .s_axis_tlast (s_tlast),
        .m_axis_tdata (m_tdata),
        .m_axis_tvalid(m_tvalid),
        .m_axis_tready(m_tready),
        .m_axis_tlast (m_tlast),
        .mode         (mode),
        .head_idx     (head_idx),
        .alpha        (alpha),
        .beta         (beta)
    );

    always #5 aclk = ~aclk;

    integer errors;
    integer checks;
    integer i, j;

    reg signed [15:0] q_buf [0:63];
    reg signed [15:0] k_buf [0:63];
    reg signed [15:0] v_buf [0:63];
    reg signed [15:0] u_q   [0:63];
    reg signed [15:0] e_buf [0:63];
    reg signed [15:0] o_ref [0:63];
    reg signed [15:0] o_got [0:63];
    integer           S     [0:4095];
    reg signed [15:0] dump_got [0:4095];
    reg signed [15:0] dump1    [0:4095];   // token 1 dump, kept for the model check
    reg signed [15:0] k1       [0:63];     // token 1 inputs, kept for the model check
    reg signed [15:0] v1       [0:63];

    reg [7:0]  alpha_b, beta_b;

    // ------------------------------------------------------------- helpers
    function signed [15:0] sat16_32;
        input signed [31:0] v;
        input integer       sh;
        reg   signed [32:0] t;
        begin
            t = {v[31], v};
            if (sh > 0)       t = (t + (33'sd1 <<< (sh - 1))) >>> sh;
            else if (sh < 0)  t = t <<< (-sh);
            if (t > 33'sd32767)      sat16_32 = 16'sh7fff;
            else if (t < -33'sd32768) sat16_32 = 16'sh8000;
            else                      sat16_32 = t[15:0];
        end
    endfunction

    task wait_idle;
        integer c;
        begin
            c = 0;
            while ((dut.state !== 4'd0) && (c < 60000)) begin
                @(posedge aclk);
                #1;
                c = c + 1;
            end
            if (c >= 60000) begin
                errors = errors + 1;
                $display("FAIL: timeout waiting for ST_IDLE (state=%0d)", dut.state);
            end
        end
    endtask

    task wait_leave_idle;
        integer c;
        begin
            c = 0;
            while ((dut.state === 4'd0) && (c < 200)) begin
                @(posedge aclk);
                #1;
                c = c + 1;
            end
        end
    endtask

    task send_token;
        integer t;
        begin
            for (t = 0; t < 64; t = t + 1) begin
                @(negedge aclk);
                s_tvalid = 1'b1;
                s_tlast  = (t == 63);
                s_tdata  = {15'd0, 1'b0, v_buf[t], k_buf[t], q_buf[t]};
                while (!s_tready) @(negedge aclk);   // tready sampled mid-cycle
                @(posedge aclk);                     // beat accepted at this edge
            end
            @(negedge aclk);
            s_tvalid = 1'b0;
            s_tlast  = 1'b0;
        end
    endtask

    task collect_o;
        integer c;
        integer n;
        begin
            n = 0;
            c = 0;
            while ((n < 64) && (c < 20000)) begin
                @(posedge aclk);
                #1;
                c = c + 1;
                if (m_tvalid) begin
                    o_got[n] = m_tdata[15:0];
                    if ((n == 63) && !m_tlast)
                        $display("WARN: m_axis_tlast not set on beat 63");
                    n = n + 1;
                end
            end
            if (n != 64) begin
                errors = errors + 1;
                $display("FAIL: decode output timeout, %0d/64 beats", n);
            end
        end
    endtask

    task collect_dump;
        integer c;
        integer n;
        begin
            n = 0;
            c = 0;
            while ((n < 4096) && (c < 40000)) begin
                @(posedge aclk);
                #1;
                c = c + 1;
                if (m_tvalid) begin
                    dump_got[n] = m_tdata[15:0];
                    if ((n == 4095) && !m_tlast)
                        $display("WARN: m_axis_tlast not set on dump beat 4095");
                    n = n + 1;
                end
            end
            if (n != 4096) begin
                errors = errors + 1;
                $display("FAIL: dump timeout, %0d/4096 beats", n);
            end
        end
    endtask

    // ------------------------------------------------- reference model
    task ref_decode;
        integer a, b;
        reg signed [63:0] bke;
        reg signed [31:0] acc;
        begin
            for (b = 0; b < 64; b = b + 1) begin
                acc = 32'sd0;
                for (a = 0; a < 64; a = a + 1)
                    acc = acc + k_buf[a] * S[a*64 + b];
                u_q[b]   = sat16_32(acc, 12);
                e_buf[b] = sat16_32(v_buf[b] - u_q[b], 0);
            end
            for (a = 0; a < 64; a = a + 1) begin
                for (b = 0; b < 64; b = b + 1) begin
                    bke = $signed({48'd0, beta}) * $signed(k_buf[a])
                          * $signed(e_buf[b]);
                    acc = ($signed({16'd0, alpha}) * S[a*64 + b]) >>> 16;
                    acc = acc + (bke >>> 31);
                    S[a*64 + b] = sat16_32(acc, 0);
                end
            end
            for (b = 0; b < 64; b = b + 1) begin
                acc = 32'sd0;
                for (a = 0; a < 64; a = a + 1)
                    acc = acc + q_buf[a] * S[a*64 + b];
                o_ref[b] = sat16_32(acc, 12);
            end
        end
    endtask

    // spec variant: e treated as Q3.13 -> beta term >>32
    task ref_decode_spec;
        integer a, b;
        reg signed [63:0] bke;
        reg signed [31:0] acc;
        begin
            for (b = 0; b < 64; b = b + 1) begin
                acc = 32'sd0;
                for (a = 0; a < 64; a = a + 1)
                    acc = acc + k_buf[a] * S[a*64 + b];
                u_q[b]   = sat16_32(acc, 12);
                e_buf[b] = sat16_32(v_buf[b] - u_q[b], 0);
            end
            for (a = 0; a < 64; a = a + 1) begin
                for (b = 0; b < 64; b = b + 1) begin
                    bke = $signed({48'd0, beta}) * $signed(k_buf[a])
                          * $signed(e_buf[b]);
                    acc = ($signed({16'd0, alpha}) * S[a*64 + b]) >>> 16;
                    acc = acc + (bke >>> 32);
                    S[a*64 + b] = sat16_32(acc, 0);
                end
            end
        end
    endtask

    // bug model: phase 4 uses the leftover k_cur (k_buf[63]) for every row
    task ref_decode_stalek;
        integer a, b;
        reg signed [63:0] bke;
        reg signed [31:0] acc;
        begin
            for (b = 0; b < 64; b = b + 1) begin
                acc = 32'sd0;
                for (a = 0; a < 64; a = a + 1)
                    acc = acc + k_buf[a] * S[a*64 + b];
                u_q[b]   = sat16_32(acc, 12);
                e_buf[b] = sat16_32(v_buf[b] - u_q[b], 0);
            end
            for (a = 0; a < 64; a = a + 1) begin
                for (b = 0; b < 64; b = b + 1) begin
                    bke = $signed({48'd0, beta}) * $signed(k_buf[63])
                          * $signed(e_buf[b]);
                    acc = ($signed({16'd0, alpha}) * S[a*64 + b]) >>> 16;
                    acc = acc + (bke >>> 31);
                    S[a*64 + b] = sat16_32(acc, 0);
                end
            end
        end
    endtask

    // ------------------------------------------------------------- main
    initial begin
        aclk      = 1'b0;
        aresetn   = 1'b0;
        s_tdata   = 64'd0;
        s_tvalid  = 1'b0;
        s_tlast   = 1'b0;
        m_tready  = 1'b1;
        mode      = 8'd3;
        head_idx  = 8'd0;
        alpha     = 16'd0;
        beta      = 16'd0;
        errors    = 0;
        checks    = 0;
        for (i = 0; i < 4096; i = i + 1) begin
            S[i] = 0;
            dump_got[i] = 16'sd0;
        end
        for (i = 0; i < 64; i = i + 1) begin
            q_buf[i] = 16'sd0;
            k_buf[i] = 16'sd0;
            v_buf[i] = 16'sd0;
            o_got[i] = 16'sd0;
            o_ref[i] = 16'sd0;
        end

        repeat (5) @(posedge aclk);
        @(negedge aclk);
        aresetn = 1'b1;
        repeat (3) @(posedge aclk);

        // ------------------------------------------------- test A: reset all
        @(negedge aclk);
        mode     = 8'd1;
        head_idx = 8'd8;
        repeat (3) @(posedge aclk);
        wait_idle;
        @(negedge aclk);
        mode = 8'd3;
        repeat (3) @(posedge aclk);

        // dump head 0 -> all zeros
        @(negedge aclk);
        mode     = 8'd2;
        head_idx = 8'd0;
        collect_dump;
        wait_idle;
        checks = checks + 1;
        begin : zero_check
            integer z;
            z = 0;
            for (i = 0; i < 4096; i = i + 1)
                if (dump_got[i] !== 16'sd0)
                    z = z + 1;
            if (z == 0)
                $display("PASS: mode=1 (all heads) then dump -> 4096/4096 state words zero");
            else begin
                errors = errors + 1;
                $display("FAIL: dump after reset has %0d/4096 non-zero words", z);
            end
        end

        // ------------------------------------------------- test B: one token
        @(negedge aclk);
        mode     = 8'd3;
        head_idx = 8'd0;
        alpha    = 16'd32768;      // 0.5  Q0.16
        beta     = 16'd65535;      // ~1.0 Q0.16
        for (i = 0; i < 64; i = i + 1) begin
            k_buf[i] = (i < 32) ? 16'sd16000 : -16'sd16000;
            v_buf[i] = (i % 2 == 0) ? 16'sd16000 : -16'sd16000;
            q_buf[i] = 16'sd100;
        end
        ref_decode;                // S is all zero, so this is the first update
        @(negedge aclk);
        mode = 8'd0;
        send_token;
        collect_o;
        @(negedge aclk);
        mode = 8'd3;

        checks = checks + 1;
        begin : o_check1
            integer e;
            e = 0;
            for (i = 0; i < 64; i = i + 1)
                if (o_got[i] !== o_ref[i]) begin
                    e = e + 1;
                    if (e <= 4)
                        $display("  o[%0d]: got %0d exp %0d", i, o_got[i], o_ref[i]);
                end
            if (e == 0)
                $display("PASS: token 1 decode output, 64/64 o[j] match reference");
            else begin
                errors = errors + 1;
                $display("FAIL: token 1 decode output, %0d/64 mismatch", e);
            end
        end

        // ------------------------------------------------- test C: dump state
        wait_idle;
        @(negedge aclk);
        mode     = 8'd2;
        head_idx = 8'd0;
        collect_dump;
        wait_idle;
        checks = checks + 1;
        begin : state_check1
            integer e;
            integer nz;
            e = 0;
            nz = 0;
            for (i = 0; i < 4096; i = i + 1) begin
                if (dump_got[i] !== S[i]) begin
                    e = e + 1;
                    if (e <= 4)
                        $display("  S[%0d][%0d]: got %0d exp %0d",
                                 i/64, i%64, dump_got[i], S[i]);
                end
                if (dump_got[i] !== 16'sd0)
                    nz = nz + 1;
            end
            if (e == 0)
                $display("PASS: token 1 state dump matches reference (%0d/4096 non-zero)", nz);
            else begin
                errors = errors + 1;
                $display("FAIL: token 1 state dump, %0d/4096 words mismatch", e);
            end
        end

        // ------------------------------- test C2: second token, non-zero state
        for (i = 0; i < 4096; i = i + 1) dump1[i] = dump_got[i];
        for (i = 0; i < 64; i = i + 1) begin
            k1[i] = k_buf[i];
            v1[i] = v_buf[i];
        end
        for (i = 0; i < 64; i = i + 1) begin
            k_buf[i] = (i < 32) ? 16'sd30 : -16'sd30;
            v_buf[i] = (i % 2 == 0) ? 16'sd300 : -16'sd300;
            q_buf[i] = 16'sd100;
        end
        ref_decode;                // S now holds token 1 state -> token 2 state
        @(negedge aclk);
        mode = 8'd0;
        send_token;
        collect_o;
        @(negedge aclk);
        mode = 8'd3;
        checks = checks + 1;
        begin : o_check2
            integer e;
            e = 0;
            for (i = 0; i < 64; i = i + 1)
                if (o_got[i] !== o_ref[i]) begin
                    e = e + 1;
                    if (e <= 4)
                        $display("  o2[%0d]: got %0d exp %0d", i, o_got[i], o_ref[i]);
                end
            if (e == 0)
                $display("PASS: token 2 (non-zero state) output, 64/64 o[j] match reference");
            else begin
                errors = errors + 1;
                $display("FAIL: token 2 output, %0d/64 mismatch", e);
            end
        end
        wait_idle;
        @(negedge aclk);
        mode     = 8'd2;
        head_idx = 8'd0;
        collect_dump;
        wait_idle;
        checks = checks + 1;
        begin : state_check2
            integer e;
            e = 0;
            for (i = 0; i < 4096; i = i + 1) begin
                if (dump_got[i] !== S[i]) begin
                    e = e + 1;
                    if (e <= 4)
                        $display("  S2[%0d][%0d]: got %0d exp %0d",
                                 i/64, i%64, dump_got[i], S[i]);
                end
            end
            if (e == 0)
                $display("PASS: token 2 state dump matches reference (4096/4096)");
            else begin
                errors = errors + 1;
                $display("FAIL: token 2 state dump, %0d/4096 words mismatch", e);
            end
        end

        // ------------------------------------- test D: Q3.13 spec comparison
        // Recompute token 1 with the documented Q3.13 delta format
        // (beta term >>32) and see whether the dump matches that instead.
        begin : spec_check
            integer m1;
            integer m2;
            integer m3;
            integer two_x;
            for (i = 0; i < 64; i = i + 1) begin
                k_buf[i] = k1[i];
                v_buf[i] = v1[i];
            end
            for (i = 0; i < 4096; i = i + 1)
                S[i] = 0;
            ref_decode_spec;
            m2 = 0;
            for (i = 0; i < 4096; i = i + 1)
                if (dump1[i] === S[i])
                    m2 = m2 + 1;
            for (i = 0; i < 4096; i = i + 1)
                S[i] = 0;
            ref_decode;
            m1 = 0;
            two_x = 0;
            for (i = 0; i < 4096; i = i + 1) begin
                if (dump1[i] === S[i]) m1 = m1 + 1;
                if ((S[i] != 0) && (dump1[i] === S[i] * 2)) two_x = two_x + 1;
            end
            for (i = 0; i < 4096; i = i + 1)
                S[i] = 0;
            ref_decode_stalek;
            m3 = 0;
            for (i = 0; i < 4096; i = i + 1)
                if (dump1[i] === S[i])
                    m3 = m3 + 1;
            $display("INFO: dump vs code model (e=Q2.14, >>31): %0d/4096 match", m1);
            $display("INFO: dump vs spec model (e=Q3.13, >>32): %0d/4096 match", m2);
            $display("INFO: dump vs stale-k model (P4 uses k_buf[63]): %0d/4096 match", m3);
            checks = checks + 1;
            if (m1 == 4096) begin
                $display("PASS: RTL matches the correct reference model bit-exactly");
            end else if (m3 == 4096) begin
                errors = errors + 1;
                $display("FAIL: phase 4 uses the stale k_cur (k_buf[63]) for every row");
            end else begin
                errors = errors + 1;
                $display("FAIL: RTL matches neither the correct nor the stale-k model");
            end
        end

        // ------------------------------------- test E: reset head 0 only
        @(negedge aclk);
        mode     = 8'd1;
        head_idx = 8'd0;
        repeat (3) @(posedge aclk);
        wait_idle;
        @(negedge aclk);
        mode = 8'd3;
        repeat (3) @(posedge aclk);
        @(negedge aclk);
        mode     = 8'd2;
        head_idx = 8'd0;
        collect_dump;
        wait_idle;
        checks = checks + 1;
        begin : reset_head_check
            integer z;
            z = 0;
            for (i = 0; i < 4096; i = i + 1)
                if (dump_got[i] !== 16'sd0)
                    z = z + 1;
            if (z == 0)
                $display("PASS: mode=1 head_idx=0 then dump -> head 0 all zero");
            else begin
                errors = errors + 1;
                $display("FAIL: reset head 0 left %0d/4096 non-zero words", z);
            end
        end

        repeat (10) @(posedge aclk);
        $display("--------------------------------------------------");
        if (errors == 0)
            $display("RESULT: tb_gdn_core PASS (%0d checks, 0 errors)", checks);
        else
            $display("RESULT: tb_gdn_core FAIL (%0d errors, %0d checks)", errors, checks);
        $display("--------------------------------------------------");
        $finish;
    end

    initial begin
        #5000000;
        $display("FAIL: global timeout");
        $finish;
    end

endmodule
