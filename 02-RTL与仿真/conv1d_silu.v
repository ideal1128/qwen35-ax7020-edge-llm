// conv1d_silu.v -- depthwise causal Conv1D (K=4) followed by SiLU
//
// One token is 6144 channels.  They arrive 4 per beat as 4 signed int16 Q2.14
// samples, so a token is 1536 beats.
//
// For every channel c:
//   y[c] = W0[c]*x_t[c] + W1[c]*x_{t-1}[c] + W2[c]*x_{t-2}[c] + W3[c]*x_{t-3}[c] + b[c]
//   out[c] = y[c] * sigmoid(y[c])
//
// History is kept as one 64-bit word per channel holding the last four samples,
// so a channel needs exactly one read and one write per token.
//
// SiLU without a divider: x is Q2.14, so the real range is [-2, +2).  sigmoid is
// tabulated at 9 uniformly spaced breakpoints (0, 0.25, ... 2.0) and linearly
// interpolated.  Breakpoint spacing in Q2.14 integers is exactly 4096, so the
// segment index is a[11:0]'s upper bits -- pure shifts, no division.
//
// Latency: 2 cycles per channel, 4 channels per beat -> 8 cycles per beat,
// 12288 cycles per token (about 74 us at 166 MHz).

`timescale 1ns / 1ps

module conv1d_silu (
    input  wire        aclk,
    input  wire        aresetn,
    input  wire [63:0] s_axis_tdata,   // 4 x int16 Q2.14
    input  wire        s_axis_tvalid,
    output wire        s_axis_tready,
    input  wire        s_axis_tlast,
    output wire [63:0] m_axis_tdata,
    output wire        m_axis_tvalid,
    output wire        m_axis_tlast,
    input  wire        m_axis_tready,
    // weight load port: 6144 entries of {b[15:0], W3, W2, W1, W0}
    input  wire        w_wr_en,
    input  wire [12:0] w_wr_addr,
    input  wire [79:0] w_wr_data,
    // 1 = bypass the conv and SiLU entirely
    input  wire        bypass
);

    // ------------------------------------------------------------ history RAM
    // 6144 channels x 64 bit = 48 KB.  Read-first: we present the address, and
    // one cycle later the old word is available while we write the new one.
    (* ram_style = "block" *) reg [63:0] hist [0:6143];

    // ------------------------------------------------------------ weight RAM
    (* ram_style = "block" *) reg [79:0] wmem [0:6143];
    always @(posedge aclk) begin
        if (w_wr_en)
            wmem[w_wr_addr] <= w_wr_data;
    end

    // ------------------------------------------------------------ sigmoid LUT
    // sigmoid at x = 0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2.0, in Q1.15.
    // Nine breakpoints, so the index needs 4 bits: seg runs 0..7 for a in
    // [0,2.0) and seg+1 runs 1..8.  Case 8 (sigmoid(2.0)=0.8808) was missing,
    // which made seg=7 fall through to the default and produce a NEGATIVE
    // interpolation slope.
    function [15:0] sig_base;
        input [3:0] seg;
        begin
            case (seg)
                4'd0: sig_base = 16'd16384;   // 0.5000
                4'd1: sig_base = 16'd18422;   // 0.5622
                4'd2: sig_base = 16'd20398;   // 0.6225
                4'd3: sig_base = 16'd22256;   // 0.6792
                4'd4: sig_base = 16'd23953;   // 0.7311
                4'd5: sig_base = 16'd25469;   // 0.7773
                4'd6: sig_base = 16'd26790;   // 0.8176
                4'd7: sig_base = 16'd27930;   // 0.8524
                4'd8: sig_base = 16'd28862;   // 0.8808  sigmoid(2.0)
                default: sig_base = 16'd28862; // a is clamped to 2.0 anyway
            endcase
        end
    endfunction

    // sigmoid of a signed Q2.14 value, returned in Q1.15
    function [15:0] sigmoid_q15;
        input signed [15:0] x;
        reg   [15:0] a;
        reg   [2:0]  seg;
        reg   [15:0] b0, b1;
        reg   [31:0] interp;
        reg   [15:0] pos;
        begin
            a = x[15] ? (16'd0 - x[15:0]) : x[15:0];   // |x| in Q2.14
            if (a >= 16'd32768) a = 16'd32767;
            seg = a[14:12];                            // 4096 per segment
            b0  = sig_base({1'b0, seg});
            b1  = sig_base({1'b0, seg} + 4'd1);
            interp = (b1 - b0) * a[11:0];
            pos = b0 + interp[23:12];
            sigmoid_q15 = x[15] ? (16'd32768 - pos) : pos;   // sig(-x) = 1 - sig(x)
        end
    endfunction

    // saturating Q2.14 product: y(Q2.14) * sig(Q1.15) -> Q2.14
    function signed [15:0] mul_q14_q15;
        input signed [15:0] y;
        input        [15:0] s;
        reg   signed [31:0] t;
        begin
            t = y * $signed({1'b0, s[14:0]});    // Q3.29
            t = t >>> 15;                        // Q2.14
            if (t > 32'sd32767)       mul_q14_q15 = 16'sd32767;
            else if (t < -32'sd32768) mul_q14_q15 = -16'sd32768;
            else                      mul_q14_q15 = t[15:0];
        end
    endfunction

    // ------------------------------------------------------------ control
    reg        busy;
    reg [1:0]  ch;              // which of the 4 channels of the current beat
    reg [1:0]  ph;              // 0 = address out, 1 = data in / compute
    reg [63:0] beat_in;
    reg        beat_last;
    reg [63:0] beat_out;
    reg        out_valid;
    reg        out_last;
    reg [12:0] base_ch;         // first channel index of the current beat
    reg        rd_en;
    reg [63:0] rd_data;
    reg [79:0] w_data;
    reg [63:0] raw_out;         // bypass path

    wire [12:0] cur_ch = base_ch + {11'd0, ch};
    wire signed [15:0] x_now = beat_in[ch*16 +: 16];

    assign s_axis_tready = ~busy;

    // channel result
    reg signed [15:0] y_conv;
    reg signed [15:0] ch_result;
    reg        [15:0] w0, w1, w2, w3, bb;
    reg signed [15:0] h1, h2, h3;
    reg signed [31:0] acc;

    always @(posedge aclk) begin
        if (!aresetn) begin
            busy      <= 1'b0;
            ch        <= 2'd0;
            ph        <= 1'b0;
            base_ch   <= 13'd0;
            beat_in   <= 64'd0;
            beat_out  <= 64'd0;
            beat_last <= 1'b0;
            out_valid <= 1'b0;
            out_last  <= 1'b0;
            rd_en     <= 1'b0;
            rd_data   <= 64'd0;
            w_data    <= 80'd0;
        end else begin
            rd_en <= 1'b0;

            // ------------------------------------------------ accept a beat
            if (s_axis_tvalid && !busy) begin
                beat_in   <= s_axis_tdata;
                beat_last <= s_axis_tlast;
                base_ch   <= 13'd0;
                ch        <= 2'd0;
                ph        <= 1'b0;
                busy      <= 1'b1;
            end

            // ------------------------------------------------ process channels
            if (busy) begin
                if (ph == 1'b0) begin
                    // phase 0: present the history and weight addresses
                    rd_data <= hist[cur_ch];
                    w_data  <= wmem[cur_ch];
                    rd_en   <= 1'b1;
                    ph      <= 1'b1;
                end else begin
                    // phase 1: data has arrived, compute and write back
                    {bb, w3, w2, w1, w0} = w_data;
                    h1 = rd_data[63:48];
                    h2 = rd_data[47:32];
                    h3 = rd_data[31:16];

                    if (bypass) begin
                        ch_result = x_now;
                    end else begin
                        acc = $signed(w0) * $signed(x_now)
                            + $signed(w1) * h1
                            + $signed(w2) * h2
                            + $signed(w3) * h3
                            + $signed(bb);
                        y_conv = acc >>> 14;          // Q2.14
                        ch_result = mul_q14_q15(y_conv, sigmoid_q15(y_conv));
                    end

                    hist[cur_ch] <= {x_now, rd_data[63:16]};
                    beat_out[ch*16 +: 16] <= ch_result;

                    if (ch == 2'd3) begin
                        ch        <= 2'd0;
                        base_ch   <= base_ch + 13'd4;
                        busy      <= 1'b0;
                        out_valid <= 1'b1;
                        out_last  <= beat_last;
                    end else begin
                        ch <= ch + 2'd1;
                        ph <= 1'b0;
                    end
                end
            end

            // ------------------------------------------------ output handshake
            if (out_valid && m_axis_tready)
                out_valid <= 1'b0;
        end
    end

    assign m_axis_tdata  = beat_out;
    assign m_axis_tvalid = out_valid;
    assign m_axis_tlast  = out_last;

endmodule
