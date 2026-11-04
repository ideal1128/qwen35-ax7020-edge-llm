// tb_conv_core.v -- end-to-end testbench for conv_core
//
// Configuration: conv bypass = 1, FWHT bypass = 0 (i.e. cfg_fwht_enable=0),
// so 6144 Q2.14 samples pass through the conv, the gearbox and the FWHT
// unchanged and are packed 16 samples per BFP8 beat.
//
// Expected: 1536 input beats -> 384 output beats, tlast on beat 383,
// no X on the output bus.
`timescale 1ns / 1ps

module tb_conv_core;

    reg          aclk;
    reg          aresetn;

    reg  [63:0]  s_tdata;
    reg          s_tvalid;
    wire         s_tready;
    reg          s_tlast;

    wire [127:0] m_tdata;
    wire         m_tvalid;
    reg          m_tready;
    wire         m_tlast;

    reg  [15:0]  cfg_channel_len;
    reg  [15:0]  cfg_conv_bypass;
    reg  [15:0]  cfg_fwht_enable;

    reg          w_wr_en;
    reg  [12:0]  w_wr_addr;
    reg  [79:0]  w_wr_data;

    conv_core dut (
        .aclk              (aclk),
        .aresetn           (aresetn),
        .s_axis_token_tdata (s_tdata),
        .s_axis_token_tvalid(s_tvalid),
        .s_axis_token_tready(s_tready),
        .s_axis_token_tlast (s_tlast),
        .m_axis_bfp_tdata  (m_tdata),
        .m_axis_bfp_tvalid (m_tvalid),
        .m_axis_bfp_tready (m_tready),
        .m_axis_bfp_tlast  (m_tlast),
        .cfg_channel_len   (cfg_channel_len),
        .cfg_conv_bypass   (cfg_conv_bypass),
        .cfg_fwht_enable   (cfg_fwht_enable),
        .w_wr_en           (w_wr_en),
        .w_wr_addr         (w_wr_addr),
        .w_wr_data         (w_wr_data)
    );

    always #5 aclk = ~aclk;

    integer errors;
    integer checks;
    integer i;

    reg [127:0] out_beats [0:511];
    integer     out_n;
    integer     last_n;
    integer     x_count;

    always @(posedge aclk) begin
        if (m_tvalid && m_tready) begin
            out_beats[out_n] <= m_tdata;
            if (m_tlast)
                last_n <= out_n;
            if (^m_tdata === 1'bx)
                x_count <= x_count + 1;
            out_n <= out_n + 1;
        end
    end

    function signed [15:0] x_val;
        input integer c;
        begin
            x_val = (c % 128) * 200 - 12000;
        end
    endfunction

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

    initial begin
        aclk             = 1'b0;
        aresetn          = 1'b0;
        s_tdata          = 64'd0;
        s_tvalid         = 1'b0;
        s_tlast          = 1'b0;
        m_tready         = 1'b1;
        cfg_channel_len  = 16'd6144;
        cfg_conv_bypass  = 16'h0001;
        cfg_fwht_enable  = 16'h0000;
        w_wr_en          = 1'b0;
        w_wr_addr        = 13'd0;
        w_wr_data        = 80'd0;
        errors           = 0;
        checks           = 0;
        out_n            = 0;
        last_n           = -1;
        x_count          = 0;
        for (i = 0; i < 512; i = i + 1) out_beats[i] = 128'd0;

        repeat (5) @(posedge aclk);
        @(negedge aclk);
        aresetn = 1'b1;
        repeat (3) @(posedge aclk);

        // BRAM history is zero after configuration in hardware
        for (i = 0; i < 6144; i = i + 1)
            dut.u_conv.hist[i] = 64'd0;

        feed_token;

        begin : wait_beats
            integer c;
            c = 0;
            while ((out_n < 384) && (c < 60000)) begin
                @(posedge aclk);
                #1;
                c = c + 1;
            end
        end

        checks = checks + 1;
        if (out_n == 384)
            $display("PASS: 1536 input beats -> 384/384 BFP8 output beats");
        else begin
            errors = errors + 1;
            $display("FAIL: expected 384 BFP8 beats, observed %0d", out_n);
        end

        checks = checks + 1;
        if (last_n == 383)
            $display("PASS: m_axis_bfp_tlast asserted on beat 383");
        else begin
            errors = errors + 1;
            $display("FAIL: tlast on beat %0d, expected 383", last_n);
        end

        checks = checks + 1;
        if (x_count == 0)
            $display("PASS: no X on m_axis_bfp_tdata");
        else begin
            errors = errors + 1;
            $display("FAIL: %0d output beats contain X", x_count);
        end

        // spot check: beat 0 packs samples 0..14, exponent from max |sample|
        checks = checks + 1;
        begin : spot_check
            integer k;
            integer amax;
            integer epos;
            integer expb;
            integer m0;
            integer a;
            amax = 0;
            for (k = 0; k < 16; k = k + 1) begin
                a = x_val(k);
                if (a < 0) a = -a;
                if (a > amax) amax = a;
            end
            epos = 0;
            for (k = 0; k < 17; k = k + 1)
                if ((amax >> k) & 1) epos = k;
            expb = epos + 134;
            m0 = $signed(out_beats[0][15:8]);
            if ((out_beats[0][7:0] == expb[7:0]) && (m0 == $signed(
                    ((x_val(0) + (1 <<< (epos - 7))) >>> (epos - 6)))))
                $display("PASS: beat 0 exponent byte = %0d, mantissa 0 = %0d (as expected)",
                         out_beats[0][7:0], m0);
            else begin
                errors = errors + 1;
                $display("FAIL: beat 0 exp byte = %0d (expected %0d), mant0 = %0d",
                         out_beats[0][7:0], expb, m0);
            end
        end

        repeat (10) @(posedge aclk);
        $display("--------------------------------------------------");
        if (errors == 0)
            $display("RESULT: tb_conv_core PASS (%0d checks, 0 errors)", checks);
        else
            $display("RESULT: tb_conv_core FAIL (%0d errors, %0d checks)", errors, checks);
        $display("--------------------------------------------------");
        $finish;
    end

    initial begin
        #3000000;
        $display("FAIL: global timeout");
        $finish;
    end

endmodule
