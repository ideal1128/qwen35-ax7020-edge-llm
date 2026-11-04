// tb_fwht64.v -- self-checking testbench for fwht64
//
// Checks:
//   1. impulse (one 1000, then 63 zeros)  -> all 64 outputs equal 125
//   2. constant vector                    -> only output 0 non-zero
//   3. impulse at index 1                 -> alternating +/- (ordering probe)
//   4. ramp vector                        -> compared to a full 64x64 reference
//   5. pseudo-random vector               -> compared to a full 64x64 reference
//   6. full-scale constant (32767)        -> saturation check
//   7. bypass=1                           -> outputs equal inputs, 1 cycle later
//   8. normal block after bypass          -> recovery check
//
// The reference is the definition z[i] = (1/8) * sum_j (-1)^popcount(i&j) a[j]
// with an arithmetic >>3 and saturation to [-32768, 32767].
`timescale 1ns / 1ps

module tb_fwht64;

    reg                aclk;
    reg                aresetn;
    reg                in_valid;
    reg signed [15:0]  in_data;
    reg                in_last;
    reg                bypass;
    wire               out_valid;
    wire signed [15:0] out_data;
    wire               out_last;

    fwht64 dut (
        .aclk      (aclk),
        .aresetn   (aresetn),
        .in_valid  (in_valid),
        .in_data   (in_data),
        .in_last   (in_last),
        .out_valid (out_valid),
        .out_data  (out_data),
        .out_last  (out_last),
        .bypass    (bypass)
    );

    always #5 aclk = ~aclk;          // 100 MHz

    integer errors;
    integer checks;
    integer i;
    integer cyc_cnt;
    integer first_in_cyc;
    integer last_in_cyc;
    integer out0_cyc;
    integer out63_cyc;
    integer n_seen;

    reg signed [15:0] vec  [0:63];
    reg signed [15:0] gotv [0:63];

    always @(posedge aclk)
        cyc_cnt <= cyc_cnt + 1;

    // ------------------------------------------------------------------ ref
    function integer popcnt;
        input integer v;
        integer t;
        begin
            t = v;
            popcnt = 0;
            while (t != 0) begin
                popcnt = popcnt + (t & 1);
                t = t >> 1;
            end
        end
    endfunction

    function integer ref_fwht;
        input integer idx;
        integer jj;
        integer acc;
        begin
            acc = 0;
            for (jj = 0; jj < 64; jj = jj + 1) begin
                if (popcnt(idx & jj) & 1)
                    acc = acc - vec[jj];
                else
                    acc = acc + vec[jj];
            end
            acc = acc >>> 3;
            if (acc > 32767)  acc = 32767;
            if (acc < -32768) acc = -32768;
            ref_fwht = acc;
        end
    endfunction

    // ----------------------------------------------------------------- tasks
    task send_block;
        integer t;
        begin
            first_in_cyc = cyc_cnt + 1;
            for (t = 0; t < 64; t = t + 1) begin
                @(negedge aclk);
                in_valid = 1'b1;
                in_data  = vec[t];
                in_last  = (t == 63);
                if (t == 63)
                    last_in_cyc = cyc_cnt + 1;
            end
            @(negedge aclk);
            in_valid = 1'b0;
            in_last  = 1'b0;
            in_data  = 16'sd0;
        end
    endtask

    task collect_block;
        integer c;
        begin
            n_seen    = 0;
            c         = 0;
            out0_cyc  = -1;
            out63_cyc = -1;
            while (n_seen < 64 && c < 5000) begin
                @(posedge aclk);
                #1;
                c = c + 1;
                if (out_valid) begin
                    if (out0_cyc < 0)
                        out0_cyc = cyc_cnt;
                    gotv[n_seen] = out_data;
                    n_seen = n_seen + 1;
                    if (out_last)
                        out63_cyc = cyc_cnt;
                end
            end
            if (n_seen != 64) begin
                errors = errors + 1;
                $display("FAIL: output timeout, only %0d of 64 beats seen", n_seen);
            end
            if (out63_cyc != out0_cyc + 63)
                $display("WARN: out_last at %0d, out_valid[0] at %0d", out63_cyc, out0_cyc);
        end
    endtask

    task compare_ref;
        input [511:0] name;
        integer e;
        integer idx;
        integer r;
        begin
            e = 0;
            for (idx = 0; idx < 64; idx = idx + 1) begin
                r = ref_fwht(idx);
                if (gotv[idx] !== r) begin
                    e = e + 1;
                    if (e <= 4)
                        $display("  idx %0d: got %0d expected %0d", idx, gotv[idx], r);
                end
            end
            checks = checks + 1;
            if (e == 0)
                $display("PASS: %0s : 64/64 outputs match the natural-order reference", name);
            else begin
                errors = errors + 1;
                $display("FAIL: %0s : %0d/64 outputs mismatch", name, e);
            end
        end
    endtask

    task compare_const;
        input [511:0] name;
        input integer first_val;
        input integer other_val;
        integer e;
        integer idx;
        begin
            e = 0;
            if (gotv[0] !== first_val) begin
                e = e + 1;
                $display("  idx 0: got %0d expected %0d", gotv[0], first_val);
            end
            for (idx = 1; idx < 64; idx = idx + 1) begin
                if (gotv[idx] !== other_val) begin
                    e = e + 1;
                    if (e <= 4)
                        $display("  idx %0d: got %0d expected %0d", idx, gotv[idx], other_val);
                end
            end
            checks = checks + 1;
            if (e == 0)
                $display("PASS: %0s", name);
            else begin
                errors = errors + 1;
                $display("FAIL: %0s : %0d/64 outputs mismatch", name, e);
            end
        end
    endtask

    // ------------------------------------------------------------------ main
    initial begin
        aclk     = 1'b0;
        aresetn  = 1'b0;
        in_valid = 1'b0;
        in_data  = 16'sd0;
        in_last  = 1'b0;
        bypass   = 1'b0;
        errors   = 0;
        checks   = 0;
        cyc_cnt  = 0;
        for (i = 0; i < 64; i = i + 1) begin
            vec[i]  = 16'sd0;
            gotv[i] = 16'sd0;
        end

        repeat (5) @(posedge aclk);
        @(negedge aclk);
        aresetn = 1'b1;
        repeat (2) @(posedge aclk);

        // ---------------------------------------------------------- test 1
        for (i = 0; i < 64; i = i + 1) vec[i] = 16'sd0;
        vec[0] = 16'sd1000;
        send_block;
        collect_block;
        compare_const("impulse 1000 at index 0 -> all 64 outputs equal 125", 125, 125);
        $display("  INFO: impulse outputs[0..7] = %0d %0d %0d %0d %0d %0d %0d %0d",
                 gotv[0], gotv[1], gotv[2], gotv[3],
                 gotv[4], gotv[5], gotv[6], gotv[7]);
        $display("  INFO: latency first-input -> first-output = %0d cycles",
                 out0_cyc - first_in_cyc);
        $display("  INFO: latency last-input  -> first-output = %0d cycles",
                 out0_cyc - last_in_cyc);

        // ---------------------------------------------------------- test 2
        for (i = 0; i < 64; i = i + 1) vec[i] = 16'sd100;
        send_block;
        collect_block;
        compare_const("constant 100 -> only output 0 non-zero (=800)", 800, 0);

        // ---------------------------------------------------------- test 3
        // ordering probe: impulse at index 1 gives (-1)^popcount(i&1) * 125
        for (i = 0; i < 64; i = i + 1) vec[i] = 16'sd0;
        vec[1] = 16'sd1000;
        send_block;
        collect_block;
        compare_ref("impulse 1000 at index 1 (ordering probe)");
        $display("  INFO: impulse@1 outputs[0..7] = %0d %0d %0d %0d %0d %0d %0d %0d",
                 gotv[0], gotv[1], gotv[2], gotv[3],
                 gotv[4], gotv[5], gotv[6], gotv[7]);

        // ---------------------------------------------------------- test 4
        for (i = 0; i < 64; i = i + 1) vec[i] = i * 211 - 6700;
        send_block;
        collect_block;
        compare_ref("ramp vector vs 64x64 reference");

        // ---------------------------------------------------------- test 5
        for (i = 0; i < 64; i = i + 1)
            vec[i] = ((i * 4051 + 12345) % 65536) - 32768;
        send_block;
        collect_block;
        compare_ref("pseudo-random vector vs 64x64 reference");

        // ---------------------------------------------------------- test 6
        for (i = 0; i < 64; i = i + 1) vec[i] = 16'sd32767;
        send_block;
        collect_block;
        compare_const("constant 32767 -> out0 saturates to 32767, others 0", 32767, 0);

        // ---------------------------------------------------------- test 7
        for (i = 0; i < 64; i = i + 1)
            vec[i] = (i * 733 - 20000);
        @(negedge aclk);
        bypass = 1'b1;
        @(negedge aclk);
        // bypass latency is 1 cycle, so drive and collect concurrently
        fork
            send_block;
            collect_block;
        join
        checks = checks + 1;
        begin : bypass_check
            integer e;
            integer idx;
            e = 0;
            for (idx = 0; idx < 64; idx = idx + 1) begin
                if (gotv[idx] !== vec[idx]) begin
                    e = e + 1;
                    if (e <= 4)
                        $display("  idx %0d: got %0d expected %0d",
                                 idx, gotv[idx], vec[idx]);
                end
            end
            if (e == 0)
                $display("PASS: bypass=1 : 64/64 inputs passed through unchanged");
            else begin
                errors = errors + 1;
                $display("FAIL: bypass=1 : %0d/64 outputs mismatch", e);
            end
        end
        @(negedge aclk);
        bypass = 1'b0;

        // ---------------------------------------------------------- test 8
        for (i = 0; i < 64; i = i + 1) vec[i] = (i * 97) - 3000;
        send_block;
        collect_block;
        compare_ref("normal block after bypass (recovery)");

        repeat (5) @(posedge aclk);
        $display("--------------------------------------------------");
        if (errors == 0)
            $display("RESULT: tb_fwht64 PASS (%0d checks, 0 errors)", checks);
        else
            $display("RESULT: tb_fwht64 FAIL (%0d errors, %0d checks)", errors, checks);
        $display("--------------------------------------------------");
        $finish;
    end

    initial begin
        #2000000;
        $display("FAIL: global timeout");
        $finish;
    end

endmodule
