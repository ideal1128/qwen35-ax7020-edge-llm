// conv_core.v -- causal Conv1D + SiLU -> 64-point FWHT -> BFP8 packer
//
// Top level for the attention front-end block.  One token is cfg_channel_len
// (default 6144) channels of Q2.14, arriving 4 per beat as a 64-bit stream.
// Output is one 128-bit beat per 16-element BFP8 block.
//
// Pipeline:
//   conv1d_silu   : depthwise causal Conv1D (K=4) + SiLU, 4 channels per beat
//   gearbox       : 64-bit (4 samples) -> 16-bit (1 sample)
//   fwht64        : 6-stage radix-2 Hadamard, >>3, saturate
//   bfp8_pack16   : shared exponent + 15 int8 mantissas per 16 samples
//
// KNOWN LIMITATION: the output side is only lightly buffered.  bfp8_pack16 holds
// its beat until m_axis_bfp_tready is seen, but fwht64 has no backpressure port,
// so a stall longer than 16 cycles can lose samples.  The intended consumer is
// an AXI DMA S2MM channel, which is always ready.  To make it robust, insert a
// small FIFO between fwht64 and bfp8_pack16.
//
// Throughput: conv1d_silu is 8 cycles per input beat (4 channels x 2 phases),
// so 1536 beats = 12288 cycles per token, about 74 us at 166 MHz.  fwht64 needs
// 320 cycles per 64-element block and there are 96 blocks, so it is the
// bottleneck at about 19200 cycles per token (~116 us at 166 MHz).

`timescale 1ns / 1ps

module conv_core (
    input  wire         aclk,
    input  wire         aresetn,

    input  wire [63:0]  s_axis_token_tdata,
    input  wire         s_axis_token_tvalid,
    output wire         s_axis_token_tready,
    input  wire         s_axis_token_tlast,

    output wire [127:0] m_axis_bfp_tdata,
    output wire         m_axis_bfp_tvalid,
    input  wire         m_axis_bfp_tready,
    output wire         m_axis_bfp_tlast,

    input  wire [15:0]  cfg_channel_len,
    input  wire [15:0]  cfg_conv_bypass,
    input  wire [15:0]  cfg_fwht_enable,

    // weight load port for the conv taps: {b, W3, W2, W1, W0} per channel
    input  wire         w_wr_en,
    input  wire [12:0]  w_wr_addr,
    input  wire [79:0]  w_wr_data
);

    // ---------------------------------------------------------------- stage 1
    wire [63:0] c1_data;
    wire        c1_valid;
    wire        c1_ready;
    wire        c1_last;

    conv1d_silu u_conv (
        .aclk           (aclk),
        .aresetn        (aresetn),
        .s_axis_tdata   (s_axis_token_tdata),
        .s_axis_tvalid  (s_axis_token_tvalid),
        .s_axis_tready  (s_axis_token_tready),
        .s_axis_tlast   (s_axis_token_tlast),
        .m_axis_tdata   (c1_data),
        .m_axis_tvalid  (c1_valid),
        .m_axis_tlast   (c1_last),
        .m_axis_tready  (c1_ready),
        .w_wr_en        (w_wr_en),
        .w_wr_addr      (w_wr_addr),
        .w_wr_data      (w_wr_data),
        .bypass         (cfg_conv_bypass[0])
    );

    // ---------------------------------------------------------------- gearbox
    // 4 samples per beat in, one sample per cycle out.
    reg [63:0] g_hold;
    reg [1:0]  g_idx;
    reg        g_busy;
    reg        g_last;
    reg        s1_valid;
    reg signed [15:0] s1_data;
    reg        s1_last;

    assign c1_ready = ~g_busy;

    always @(posedge aclk) begin
        if (!aresetn) begin
            g_hold   <= 64'd0;
            g_idx    <= 2'd0;
            g_busy   <= 1'b0;
            g_last   <= 1'b0;
            s1_valid <= 1'b0;
            s1_data  <= 16'sd0;
            s1_last  <= 1'b0;
        end else begin
            s1_valid <= 1'b0;

            if (!g_busy) begin
                if (c1_valid) begin
                    g_hold <= c1_data;
                    g_last <= c1_last;
                    g_idx  <= 2'd0;
                    g_busy <= 1'b1;
                end
            end else begin
                s1_data  <= g_hold[g_idx*16 +: 16];
                s1_valid <= 1'b1;
                s1_last  <= g_last && (g_idx == 2'd3);
                if (g_idx == 2'd3)
                    g_busy <= 1'b0;
                else
                    g_idx <= g_idx + 2'd1;
            end
        end
    end

    // ---------------------------------------------------------------- stage 2
    wire signed [15:0] s2_data;
    wire               s2_valid;
    wire               s2_last;

    fwht64 u_fwht (
        .aclk     (aclk),
        .aresetn  (aresetn),
        .in_valid (s1_valid),
        .in_data  (s1_data),
        .in_last  (s1_last),
        .out_valid(s2_valid),
        .out_data (s2_data),
        .out_last (s2_last),
        .bypass   (~cfg_fwht_enable[0])
    );

    // ---------------------------------------------------------------- stage 3
    bfp8_pack16 u_pack (
        .aclk      (aclk),
        .aresetn   (aresetn),
        .in_valid  (s2_valid),
        .in_data   (s2_data),
        .in_last   (s2_last),
        .out_valid (m_axis_bfp_tvalid),
        .out_data  (m_axis_bfp_tdata),
        .out_last  (m_axis_bfp_tlast),
        .out_ready (m_axis_bfp_tready)
    );

endmodule
