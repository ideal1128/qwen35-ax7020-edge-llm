// tb_bfp8_pack16.v -- self-checking testbench for bfp8_pack16
//
// The reference is the documented algorithm:
//   amax    = max |z[i]|  (i = 0..15)
//   E       = highest set bit of amax (0 when amax == 0)
//   exp     = (amax == 0) ? 0 : E + 134
//   mant[i] = clamp(round(z[i] / 2^(E-6)), -127, +127)   for i = 0..14
//   beat    = {mant14, ..., mant0, exp}
//   mantissa 15 is discarded by design (128-bit format has 15 mantissa bytes)
//
// Cases: identical values, increasing magnitudes, all zeros, +full-scale,
// -full-scale, pseudo-random, and an out_ready backpressure check.
`timescale 1ns / 1ps

module tb_bfp8_pack16;

    reg                aclk;
    reg                aresetn;
    reg                in_valid;
    reg signed [15:0]  in_data;
    reg                in_last;
    reg                out_ready;
    wire               out_valid;
    wire [127:0]       out_data;
    wire               out_last;

    bfp8_pack16 dut (
        .aclk      (aclk),
        .aresetn   (aresetn),
        .in_valid  (in_valid),
        .in_data   (in_data),
        .in_last   (in_last),
        .out_valid (out_valid),
        .out_data  (out_data),
        .out_last  (out_last),
        .out_ready (out_ready)
    );

    always #5 aclk = ~aclk;

    integer errors;
    integer checks;
    integer i;
    integer cyc_cnt;

    reg signed [15:0] vec [0:15];

    always @(posedge aclk)
        cyc_cnt <= cyc_cnt + 1;

    // ------------------------------------------------------------------ ref
    function integer abs_i;
        input integer x;
        begin
            abs_i = (x < 0) ? -x : x;
        end
    endfunction

    function integer lop17;
        input integer v;
        integer k;
        begin
            lop17 = 0;
            for (k = 0; k < 17; k = k + 1)
                if ((v >> k) & 1)
                    lop17 = k;
        end
    endfunction

    function [127:0] ref_pack;
        input integer dummy;
        integer idx;
        integer amax;
        integer epos;
        integer sh;
        integer t;
        reg [127:0] w;
        begin
            amax = 0;
            for (idx = 0; idx < 16; idx = idx + 1)
                if (abs_i(vec[idx]) > amax)
                    amax = abs_i(vec[idx]);

            epos = lop17(amax);
            sh   = epos - 6;

            w[7:0] = (amax == 0) ? 8'd0 : (epos + 134);
            for (idx = 0; idx < 15; idx = idx + 1) begin
                if (amax == 0) begin
                    w[8 + idx*8 +: 8] = 8'd0;
                end else begin
                    if (sh > 0) begin
                        t = vec[idx] + (1 <<< (sh - 1));
                        t = t >>> sh;
                    end else if (sh < 0) begin
                        t = vec[idx] <<< (-sh);
                    end else begin
                        t = vec[idx];
                    end
                    if (t > 127)  t = 127;
                    if (t < -127) t = -127;
                    w[8 + idx*8 +: 8] = t[7:0];
                end
            end
            ref_pack = w;
        end
    endfunction

    // ----------------------------------------------------------------- tasks
    task send16;
        input lastflag;
        integer t;
        begin
            for (t = 0; t < 16; t = t + 1) begin
                @(negedge aclk);
                in_valid = 1'b1;
                in_data  = vec[t];
                in_last  = (t == 15) ? lastflag : 1'b0;
            end
            @(negedge aclk);
            in_valid = 1'b0;
            in_last  = 1'b0;
        end
    endtask

    task wait_out;
        integer c;
        begin
            c = 0;
            @(posedge aclk);
            #1;
            while (!out_valid && c < 200) begin
                @(posedge aclk);
                #1;
                c = c + 1;
            end
        end
    endtask

    task check_case;
        input [511:0] name;
        input         expect_last;
        reg   [127:0] ref;
        integer e;
        integer idx;
        begin
            ref = ref_pack(0);
            e = 0;
            if (out_data !== ref) begin
                e = e + 1;
                $display("  got %032h", out_data);
                $display("  exp %032h", ref);
                $display("  exp byte got %0d exp %0d", out_data[7:0], ref[7:0]);
                for (idx = 0; idx < 15; idx = idx + 1)
                    if (out_data[8+idx*8 +: 8] !== ref[8+idx*8 +: 8])
                        $display("  mant %0d: got %0d exp %0d", idx,
                                 $signed(out_data[8+idx*8 +: 8]),
                                 $signed(ref[8+idx*8 +: 8]));
            end
            if (out_last !== expect_last) begin
                e = e + 1;
                $display("  out_last got %0b exp %0b", out_last, expect_last);
            end
            checks = checks + 1;
            if (e == 0)
                $display("PASS: %0s (exp=%0d, mant0=%0d)", name, out_data[7:0],
                         $signed(out_data[15:8]));
            else begin
                errors = errors + 1;
                $display("FAIL: %0s", name);
            end
        end
    endtask

    // ------------------------------------------------------------------ main
    initial begin
        aclk      = 1'b0;
        aresetn   = 1'b0;
        in_valid  = 1'b0;
        in_data   = 16'sd0;
        in_last   = 1'b0;
        out_ready = 1'b1;
        errors    = 0;
        checks    = 0;
        cyc_cnt   = 0;
        for (i = 0; i < 16; i = i + 1) vec[i] = 16'sd0;

        repeat (5) @(posedge aclk);
        @(negedge aclk);
        aresetn = 1'b1;
        repeat (2) @(posedge aclk);

        // ---------------------------------------------------- 1: identical
        for (i = 0; i < 16; i = i + 1) vec[i] = 16'sd1000;
        send16(1'b0);
        wait_out;
        check_case("16 identical values 1000", 1'b0);
        @(negedge aclk);
        out_ready = 1'b1;

        // ---------------------------------------------------- 2: increasing
        for (i = 0; i < 16; i = i + 1) vec[i] = (i + 1) * 1000;
        send16(1'b1);
        wait_out;
        check_case("16 increasing magnitudes 1000..16000", 1'b1);
        begin : mono_check
            integer k;
            integer e2;
            e2 = 0;
            for (k = 1; k < 15; k = k + 1)
                if ($signed(out_data[8+k*8 +: 8]) < $signed(out_data[8+(k-1)*8 +: 8]))
                    e2 = e2 + 1;
            checks = checks + 1;
            if (e2 == 0)
                $display("PASS: mantissas monotone non-decreasing, largest = %0d",
                         $signed(out_data[8+14*8 +: 8]));
            else begin
                errors = errors + 1;
                $display("FAIL: mantissas not monotone (%0d inversions)", e2);
            end
        end

        // ---------------------------------------------------- 3: all zeros
        for (i = 0; i < 16; i = i + 1) vec[i] = 16'sd0;
        send16(1'b0);
        wait_out;
        check_case("16 zeros -> exponent 0, mantissas 0", 1'b0);

        // ---------------------------------------------------- 4: +full scale
        for (i = 0; i < 16; i = i + 1) vec[i] = 16'sd32767;
        send16(1'b0);
        wait_out;
        check_case("16 x +32767 -> mantissa clamped to 127", 1'b0);

        // ---------------------------------------------------- 5: -full scale
        for (i = 0; i < 16; i = i + 1) vec[i] = -16'sd32768;
        send16(1'b0);
        wait_out;
        check_case("16 x -32768 -> no abs overflow", 1'b0);

        // ---------------------------------------------------- 6: random
        for (i = 0; i < 16; i = i + 1)
            vec[i] = ((i * 8191 + 4321) % 65536) - 32768;
        send16(1'b1);
        wait_out;
        check_case("pseudo-random block", 1'b1);

        // ---------------------------------------------------- 7: backpressure
        for (i = 0; i < 16; i = i + 1) vec[i] = (i * 300) - 2000;
        out_ready = 1'b0;
        send16(1'b0);
        wait_out;
        begin : bp_check
            reg [127:0] hold;
            integer e3;
            e3 = 0;
            hold = out_data;
            for (i = 0; i < 8; i = i + 1) begin
                @(posedge aclk);
                #1;
                if (!out_valid) e3 = e3 + 1;
                if (out_data !== hold) e3 = e3 + 1;
            end
            checks = checks + 1;
            if (e3 == 0)
                $display("PASS: out_valid/data held stable for 8 cycles with out_ready=0");
            else begin
                errors = errors + 1;
                $display("FAIL: backpressure: %0d violations", e3);
            end
        end
        @(negedge aclk);
        out_ready = 1'b1;
        @(posedge aclk);
        #1;
        checks = checks + 1;
        if (out_valid === 1'b0)
            $display("PASS: beat consumed when out_ready asserted");
        else begin
            errors = errors + 1;
            $display("FAIL: out_valid still high after out_ready");
        end

        repeat (5) @(posedge aclk);
        $display("--------------------------------------------------");
        if (errors == 0)
            $display("RESULT: tb_bfp8_pack16 PASS (%0d checks, 0 errors)", checks);
        else
            $display("RESULT: tb_bfp8_pack16 FAIL (%0d errors, %0d checks)", errors, checks);
        $display("--------------------------------------------------");
        $finish;
    end

    initial begin
        #2000000;
        $display("FAIL: global timeout");
        $finish;
    end

endmodule
