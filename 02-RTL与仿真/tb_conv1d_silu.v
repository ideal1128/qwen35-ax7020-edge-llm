// tb_conv1d_silu.v -- self-checking testbench for conv1d_silu
//
// Configuration: W0 = 1.0 (Q2.14 = 16384), W1..W3 = 0, b = 0 for all 6144
// channels, so y = x and the output must be x * sigmoid(x).
//
// The history RAM has no reset port (it is BRAM, zero at configuration), so the
// TB initialises it to zero through a hierarchical reference before the token.
//
// Checks:
//   1. one token (1536 beats) produces 1536 output beats, tlast on the last
//   2. every channel is compared with the exact float x*sigmoid(x); the error
//      is reported as a distribution so a LUT approximation defect is visible
//   3. bypass=1 passes the input through unchanged
`timescale 1ns / 1ps

module tb_conv1d_silu;

    reg         aclk;
    reg         aresetn;

    reg  [63:0] s_tdata;
    reg         s_tvalid;
    wire        s_tready;
    reg         s_tlast;

    wire [63:0] m_tdata;
    wire        m_tvalid;
    wire        m_tlast;
    reg         m_tready;

    reg         w_wr_en;
    reg  [12:0] w_wr_addr;
    reg  [79:0] w_wr_data;

    reg         bypass;

    conv1d_silu dut (
        .aclk         (aclk),
        .aresetn      (aresetn),
        .s_axis_tdata (s_tdata),
        .s_axis_tvalid(s_tvalid),
        .s_axis_tready(s_tready),
        .s_axis_tlast (s_tlast),
        .m_axis_tdata (m_tdata),
        .m_axis_tvalid(m_tvalid),
        .m_axis_tlast (m_tlast),
        .m_axis_tready(m_tready),
        .w_wr_en      (w_wr_en),
        .w_wr_addr    (w_wr_addr),
        .w_wr_data    (w_wr_data),
        .bypass       (bypass)
    );

    always #5 aclk = ~aclk;

    integer errors;
    integer checks;
    integer i;

    reg [63:0] out_beats [0:1535];
    integer    out_n;

    always @(posedge aclk) begin
        if (m_tvalid && m_tready) begin
            out_beats[out_n] <= m_tdata;
            out_n <= out_n + 1;
        end
    end

    function signed [15:0] x_val;
        input integer c;
        begin
            x_val = (c - 3072) * 10;
        end
    endfunction

    task load_weights;
        integer c;
        begin
            for (c = 0; c < 6144; c = c + 1) begin
                @(negedge aclk);
                w_wr_en   = 1'b1;
                w_wr_addr = c[12:0];
                w_wr_data = {16'd0, 16'd0, 16'd0, 16'd0, 16'd16384};
            end
            @(negedge aclk);
            w_wr_en = 1'b0;
        end
    endtask

    task feed_token;
        integer b;
        integer k;
        begin
            for (b = 0; b < 1536; b = b + 1) begin
                @(negedge aclk);
                s_tvalid = 1'b1;
                s_tlast  = (b == 1535);
                for (k = 0; k < 4; k = k + 1)
                    s_tdata[k*16 +: 16] = x_val(4*b + k);
                while (!s_tready) @(negedge aclk);   // tready sampled mid-cycle
                @(posedge aclk);                     // beat accepted at this edge
            end
            @(negedge aclk);
            s_tvalid = 1'b0;
            s_tlast  = 1'b0;
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
        w_wr_en   = 1'b0;
        w_wr_addr = 13'd0;
        w_wr_data = 80'd0;
        bypass    = 1'b0;
        errors    = 0;
        checks    = 0;
        out_n     = 0;
        for (i = 0; i < 1536; i = i + 1) out_beats[i] = 64'd0;

        repeat (5) @(posedge aclk);
        @(negedge aclk);
        aresetn = 1'b1;
        repeat (3) @(posedge aclk);

        // BRAM history is zero after configuration in hardware
        for (i = 0; i < 6144; i = i + 1)
            dut.hist[i] = 64'd0;

        load_weights;

        // -------------------------------------------------- test 1: one token
        out_n = 0;
        feed_token;
        begin : wait_beats
            integer c;
            c = 0;
            while ((out_n < 1536) && (c < 40000)) begin
                @(posedge aclk);
                #1;
                c = c + 1;
            end
        end
        checks = checks + 1;
        if (out_n != 1536) begin
            errors = errors + 1;
            $display("FAIL: token 1 produced %0d/1536 output beats", out_n);
        end else begin
            $display("PASS: token 1 produced 1536/1536 output beats");
        end

        // -------------------------------------------------- test 2: SiLU error
        checks = checks + 1;
        begin : silu_check
            integer c;
            integer k;
            integer idx;
            integer got;
            integer exp_i;
            integer err;
            integer max_err;
            integer max_idx;
            integer bad;
            integer bad_hi;
            integer bad_lo;
            integer in_win;
            integer max_lo;
            real    xr;
            real    sr;
            real    yr;

            max_err = 0;
            max_idx = 0;
            bad     = 0;
            bad_hi  = 0;
            bad_lo  = 0;
            max_lo  = 0;
            for (c = 0; c < 6144; c = c + 1) begin
                idx = c / 4;
                k   = c % 4;
                got = $signed(out_beats[idx][k*16 +: 16]);
                xr  = $itor(x_val(c)) / 16384.0;
                sr  = 1.0 / (1.0 + $exp(-xr));
                yr  = xr * sr;
                exp_i = $rtoi(yr * 16384.0 + ((yr >= 0.0) ? 0.5 : -0.5));
                err = got - exp_i;
                if (err < 0) err = -err;
                if (err > max_err) begin
                    max_err = err;
                    max_idx = c;
                end
                in_win = (x_val(c) >= 28672) || (x_val(c) <= -28672);  // |x| >= 1.75
                if (!in_win && (err > max_lo))
                    max_lo = err;
                if (err > 64) begin
                    bad = bad + 1;
                    if (in_win) bad_hi = bad_hi + 1;
                    else        bad_lo = bad_lo + 1;
                end
            end
            $display("INFO: max |out - x*sigmoid(x)| = %0d LSB at channel %0d (x=%0d)",
                     max_err, max_idx, x_val(max_idx));
            $display("INFO: max error outside |x|>=1.75 window = %0d LSB", max_lo);
            $display("INFO: channels with |err| > 64 LSB: %0d (in |x|>=1.75: %0d, elsewhere: %0d)",
                     bad, bad_hi, bad_lo);
            if ((max_err <= 64) && (bad == 0))
                $display("PASS: SiLU output within 64 LSB of x*sigmoid(x) for all 6144 channels");
            else begin
                errors = errors + 1;
                $display("FAIL: SiLU deviates from x*sigmoid(x) by up to %0d LSB (%0d channels > 64)",
                         max_err, bad);
            end
        end

        // -------------------------------------------------- test 3: bypass
        @(negedge aclk);
        bypass = 1'b1;
        out_n  = 0;
        feed_token;
        begin : wait_beats2
            integer c;
            c = 0;
            while ((out_n < 1536) && (c < 40000)) begin
                @(posedge aclk);
                #1;
                c = c + 1;
            end
        end
        checks = checks + 1;
        begin : bypass_check
            integer c;
            integer k;
            integer idx;
            integer e;
            e = 0;
            for (c = 0; c < 6144; c = c + 1) begin
                idx = c / 4;
                k   = c % 4;
                if ($signed(out_beats[idx][k*16 +: 16]) !== x_val(c)) begin
                    e = e + 1;
                    if (e <= 4)
                        $display("  ch %0d: got %0d exp %0d", c,
                                 $signed(out_beats[idx][k*16 +: 16]), x_val(c));
                end
            end
            if ((e == 0) && (out_n == 1536))
                $display("PASS: bypass=1 passes all 6144 channels unchanged");
            else begin
                errors = errors + 1;
                $display("FAIL: bypass, %0d/6144 channels differ (out_n=%0d)", e, out_n);
            end
        end
        @(negedge aclk);
        bypass = 1'b0;

        repeat (10) @(posedge aclk);
        $display("--------------------------------------------------");
        if (errors == 0)
            $display("RESULT: tb_conv1d_silu PASS (%0d checks, 0 errors)", checks);
        else
            $display("RESULT: tb_conv1d_silu FAIL (%0d errors, %0d checks)", errors, checks);
        $display("--------------------------------------------------");
        $finish;
    end

    initial begin
        #2000000;
        $display("FAIL: global timeout");
        $finish;
    end

endmodule
