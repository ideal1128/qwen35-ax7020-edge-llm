// bfp8_pack16.v -- BFP8 block quantiser and packer (one 128-bit beat per block)
//
// 16 signed 16-bit samples in, one 128-bit beat out:
//   out_data[7:0]     = exponent byte, biased with 134 (value = 2^(byte-134))
//   out_data[15:8]    = mantissa 0
//   out_data[23:16]   = mantissa 1
//   ...
//   out_data[127:120] = mantissa 14
//
// Mantissa 15 is DISCARDED: the 128-bit output format only has room for one
// exponent byte plus 15 mantissa bytes.  This is a documented choice forced by
// the format width.
//
// Numerics, per 16-element block:
//   amax    = max |z[i]|
//   E       = position of the highest set bit of amax   (0 when amax == 0)
//   mant[i] = clamp( round( z[i] / 2^(E-6) ), -127, +127 )
//
// Latency: 1 cycle to fill the block buffer + 1 cycle to quantise = the beat is
// presented one cycle after the 16th sample is accepted.

`timescale 1ns / 1ps

module bfp8_pack16 (
    input  wire                aclk,
    input  wire                aresetn,
    input  wire                in_valid,
    input  wire signed [15:0]  in_data,
    input  wire                in_last,
    output reg                 out_valid,
    output reg  [127:0]        out_data,
    output reg                 out_last,
    input  wire                out_ready
);

    // ---------------------------------------------------------------- storage
    reg signed [15:0] samp [0:15];
    reg        [3:0]  cnt;
    reg               last_q;
    reg               have_beat;

    // ---------------------------------------------------------------- quantise
    // |x| in 17 bits so that -32768 does not overflow.
    function [16:0] abs17;
        input signed [15:0] x;
        begin
            // NOTE: $unsigned(x) zero-extends in a 17-bit context, so negating it
            // yields 65536+|x| and wraps.  Sign-extend explicitly first.
            abs17 = x[15] ? (17'd0 - {1'b1, x[15:0]}) : {1'b0, x[15:0]};
        end
    endfunction

    // z / 2^sh with round-to-nearest, saturated to int8.  sh may be negative,
    // in which case the value is shifted left.
    function signed [7:0] quant;
        input signed [15:0] z;
        input integer       sh;
        reg   signed [31:0] t;
        begin
            if (sh > 0) begin
                t = $signed(z) + (32'sd1 <<< (sh - 1));   // round half up
                t = t >>> sh;
            end else if (sh < 0) begin
                t = $signed(z) <<< (-sh);
            end else begin
                t = $signed(z);
            end
            if (t > 32'sd127)       quant = 8'sd127;
            else if (t < -32'sd127) quant = -8'sd127;
            else                    quant = t[7:0];
        end
    endfunction

    // leading-one position of a 17-bit magnitude, 0 when the input is zero
    function integer lop;
        input [16:0] v;
        integer k;
        begin
            lop = 0;
            for (k = 0; k < 17; k = k + 1)
                if (v[k]) lop = k;
        end
    endfunction

    // ---------------------------------------------------------------- helpers
    integer i;
    reg [16:0] amax;
    integer    e_pos;
    integer    sh;
    reg [7:0]  exp_byte;
    reg [127:0] packed_bits;

    always @(posedge aclk) begin
        if (!aresetn) begin
            cnt       <= 4'd0;
            last_q    <= 1'b0;
            have_beat <= 1'b0;
            out_valid <= 1'b0;
            out_data  <= 128'd0;
            out_last  <= 1'b0;
            for (i = 0; i < 16; i = i + 1)
                samp[i] <= 16'sd0;
        end else begin
            // ------------------------------------------------ fill the block
            if (in_valid) begin
                samp[cnt] <= in_data;
                if (in_last)
                    last_q <= 1'b1;
                if (cnt == 4'd15) begin
                    cnt       <= 4'd0;
                    have_beat <= 1'b1;
                end else begin
                    cnt <= cnt + 4'd1;
                end
            end

            // ------------------------------------------------ quantise and pack
            if (have_beat) begin
                have_beat <= 1'b0;

                amax = 17'd0;
                for (i = 0; i < 16; i = i + 1) begin
                    if (abs17(samp[i]) > amax)
                        amax = abs17(samp[i]);
                end

                e_pos    = lop(amax);
                sh       = e_pos - 6;
                exp_byte = (amax == 17'd0) ? 8'd0 : (e_pos + 134);

                packed_bits[7:0] = exp_byte;
                for (i = 0; i < 15; i = i + 1) begin
                    packed_bits[8 + i*8 +: 8] =
                        (amax == 17'd0) ? 8'd0 : quant(samp[i], sh);
                end

                out_data  <= packed_bits;
                out_valid <= 1'b1;
                out_last  <= last_q;
                last_q    <= 1'b0;
            end

            // ------------------------------------------------ handshake
            if (out_valid && out_ready)
                out_valid <= 1'b0;
        end
    end

endmodule
