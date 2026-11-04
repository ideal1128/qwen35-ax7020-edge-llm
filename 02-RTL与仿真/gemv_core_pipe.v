// gemv_core.v -- BFP8 block dot-product engine, deeply pipelined for 166.67 MHz.
//
// Interface is minimal: one AXI4-Stream slave and one AXI4-Stream master, no AXI-Lite.
// Configuration travels in the first beat of the stream (the "header").
//
// Stream format (one GEMV):
//   1) header  1 beat   : bits[15:0]  = x_beats
//                         bits[23:16] = beats_per_row - 1
//                         bits[30:24] = exp_beats_per_row - 1
//                         bit[31]     = dequant_en
//                                       1: on-chip 2^(e-134) alignment + whole-row accumulation,
//                                          one finished int32 y[m] per row
//                                       0: legacy mode, one raw int32 block dot product per block
//   2) x       x_beats  : 8 int16 per beat (128 bit)
//   3) when dequant_en == 1:
//        at the start of every row: exp_beats beats of exponents (16 uint8 per beat)
//        immediately followed by  : beats_per_row beats of weights (16 int8 per beat)
//      when dequant_en == 0:
//        weights only (16 int8 per beat), s_axis_tlast on the final beat
//
// Output:
//   dequant_en == 1: one finished int32 y[m] per row
//   dequant_en == 0: one raw int32 block dot product per 64-element block
//
// ---------------------------------------------------------------------------
// Pipeline map (all stages advance one beat per clock unless noted)
//   S1  v1 : capture x_r (x RAM read data), w_r, beat context c1
//   S2  v2 : p_r[i]  = w_r[i] * x_r[i]                (16 x int8*int16)
//   S3  v3 : q0..q3  = 4-way partial sums of p_r
//   S4  v4 : q01, q23 = 8-way partial sums
//   S5  v5 : sum_r   = q01 + q23                      (32-bit beat partial dot)
//   S6  v6 : block end only -- latch bd_q = acc + sum_r (the int32 block dot)
//   S7  v7 : latch exp_q = row_exp[toggle][blk]        (register the exponent read)
//   S8  v8 : latch shift_q = 20 + exp_q - 134          (BFP8 fixed-point alignment)
//   S9  v9 : latch shifted_q = barrel_shift(bd_q, shift_q)
//   S10 v10: row_acc_64 <= row_acc_64 + shifted_q ; emit the finished row beat
//
// Rationale: the old design did exponent RAM read -> shift amount -> 64-bit barrel
// shifter -> 64-bit add -> output RAM write inside ONE 5 ns period (10.348 ns,
// 24 logic levels).  Splitting that chain across S6..S10 gives every one of those
// steps its own clock period.
//
// Rounding bias: the +2^19 rounding offset is folded into the row accumulator
// initial value (row_acc_64 starts each row at 2^19 instead of 0), so the final
// conversion is a pure bit slice row_acc_64[63:20].  This is arithmetically
// identical to ((row_acc_64 + 2^19) >>> 20) because a 64-bit arithmetic right
// shift by 20 is exactly bits [63:20] of the (already biased) value.
//
// Saturating BFP8 exponent bug fix (see the "shift > 63" comment near SAT_DECIDE):
// the legacy code forced shifted_64 = 0 whenever shift_64 > 63, silently dropping
// every block whose exponent byte e was greater than 177.  The software reference
// computes ldexpf(1.0f, e-134), which saturates to +inf for e >= 262 and is a huge
// finite value below that.  We reproduce the +inf behaviour by saturating the row
// accumulator to +/-(2^63-1) and keeping it saturated for the remainder of the row
// (an infinity absorbs any further finite term).  The 32-bit output of a saturated
// row is clamped to INT32_MAX / INT32_MIN, because (2^63-1)/2^20 does not fit in
// int32.  Non-saturated rows keep the legacy truncating conversion, bit-exact with
// the software reference.

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
               ST_EXP   = 3'd2,
               ST_RUN   = 3'd3,
               ST_DRAIN = 3'd4;

    localparam signed [63:0] ACC_MAX64 = 64'sh7FFF_FFFF_FFFF_FFFF;  // +(2^63-1)
    localparam signed [63:0] ACC_MIN64 = 64'sh8000_0000_0000_0001;  // -(2^63-1)
    localparam signed [63:0] ACC_BIAS  = 64'sd524288;              // 2^19 rounding bias

    reg [2:0]  state;
    reg [15:0] x_last;          // x_beats - 1
    reg [7:0]  bpr_m1;
    reg        dequant_en;
    reg [6:0]  exp_beats_m1;
    reg [6:0]  exp_cnt;
    reg [15:0] x_load_cnt;
    reg [127:0] x_hold;
    reg [7:0]  pos;
    reg        row_toggle;      // ping-pong row exponent buffer select

    // ---------------------------------------------------------------- ping-pong row exponents
    reg [7:0] row_exp [0:1][0:127];

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

    // ---------------------------------------------------------------- S1..S5 pipeline
    reg        v1, v2, v3, v4, v5, v6, v7, v8, v9, v10;
    // c: {s_axis_tlast, is_row_end, row_toggle, pos[7:0]} (11 bits)
    reg [10:0] c1, c2, c3, c4, c5, c6, c7, c8, c9, c10;
    reg [255:0] x_r;
    reg [127:0] w_r;
    reg signed [31:0] p_r [0:ELEM_PER_BEAT-1];
    reg signed [31:0] q0, q1, q2, q3;
    reg signed [31:0] q01, q23;
    reg signed [31:0] sum_r;
    reg signed [31:0] acc;

    wire rst_n  = aresetn & ~dbg_rst;
    wire s_fire = s_axis_tvalid & s_axis_tready;
    assign s_axis_tready = ((state == ST_HDR) || (state == ST_LOADX) ||
                            (state == ST_EXP) || (state == ST_RUN)) && !m_axis_tvalid;

    integer i;
    always @(posedge aclk) begin
        if (!rst_n) begin
            v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0; v4 <= 1'b0; v5 <= 1'b0;
            c1 <= 11'd0; c2 <= 11'd0; c3 <= 11'd0; c4 <= 11'd0; c5 <= 11'd0;
            x_r <= 256'd0; w_r <= 128'd0;
            q0 <= 0; q1 <= 0; q2 <= 0; q3 <= 0;
            q01 <= 0; q23 <= 0;
            sum_r <= 0;
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
        end
    end

    // ---------------------------------------------------------------- S5 context decode
    wire       c5_gemv_last = c5[10];
    wire       c5_row_last  = c5[9];
    wire [7:0] c5_pos       = c5[7:0];
    wire [1:0] c5_sub       = c5_pos[1:0];
    wire       blk_end5     = (c5_sub == BEATS_PER_BLOCK - 1);

    // The 32-bit beat dot product, accumulated over the BEATS_PER_BLOCK beats of a block.
    wire signed [31:0] dot_nx = acc + sum_r;

    // ---------------------------------------------------------------- S6: block dot capture
    // Only one cycle per block matters here, so this stage holds just the block dot
    // product plus the beat context needed by the dequant stages.
    reg signed [31:0] bd_q;         // int32 block dot product
    reg signed [8:0]  shift_q;      // 20 + e - 134
    reg [7:0]         exp_q;

    always @(posedge aclk) begin
        if (!rst_n) begin
            v6    <= 1'b0;
            c6    <= 11'd0;
            bd_q  <= 32'sd0;
            exp_q <= 8'd134;
        end else begin
            v6 <= v5 & blk_end5;
            c6 <= c5;
            if (v5 & blk_end5) bd_q <= dot_nx;
            // S7: register the row exponent RAM read (was a raw async read on the critical path)
            if (v6) exp_q <= row_exp[c6[8]][c6[7:2]];
        end
    end

    wire       c6_gemv_last = c6[10];
    wire       c6_row_last  = c6[9];
    wire [7:0] c6_pos       = c6[7:0];
    wire [5:0] c6_blk       = c6_pos[7:2];

    always @(posedge aclk) begin
        if (!rst_n) begin
            v7      <= 1'b0;
            c7      <= 11'd0;
            shift_q <= 9'sd0;
        end else begin
            v7 <= v6;
            c7 <= c6;
            // S8: BFP8 fixed-point alignment amount.  base point is bit 20.
            // exp_q was registered out of the exponent RAM one cycle earlier, so this
            // 9-bit add no longer sits behind the 128:1 exponent read mux.
            if (v7) shift_q <= 9'sd20 + ($signed({1'b0, exp_q}) - 9'sd134);
        end
    end

    // ---------------------------------------------------------------- S9: 64-bit barrel shift
    wire signed [63:0] ext_q   = {{32{bd_q[31]}}, bd_q};   // sign-extend the int32 block dot
    wire        [5:0]  ls_amt  = shift_q[5:0];
    wire        [8:0]  neg_mag = 9'd0 - shift_q;           // |shift| for the right-shift case
    // Ext is sign-extended in bits 63:32, so any right shift of 32 or more collapses to
    // the sign bit.  Clamping at 32 keeps the barrel shifter at six levels and is exactly
    // equivalent to the legacy "all-ones / all-zeros when -shift >= 63" special case.
    wire        [5:0]  rs_amt  = (neg_mag >= 9'd32) ? 6'd32 : neg_mag[5:0];
    wire signed [63:0] shl     = ext_q <<  ls_amt;
    wire signed [63:0] shr     = ext_q >>> rs_amt;
    wire signed [63:0] shifted_nx = shift_q[8] ? shr : shl;

    // SAT_DECIDE: a block with e > 177 gives shift_q > 63.  The legacy code produced
    // zero here (block silently dropped); we flag saturation instead.
    wire sat_nx      = (shift_q > 9'sd63);
    wire sat_sign_nx = bd_q[31];

    reg signed [63:0] shifted_q;
    reg               sat_q;
    reg               sat_sign_q;

    always @(posedge aclk) begin
        if (!rst_n) begin
            v8         <= 1'b0;
            c8         <= 11'd0;
            shifted_q  <= 64'sd0;
            sat_q      <= 1'b0;
            sat_sign_q <= 1'b0;
        end else begin
            v8 <= v7;
            c8 <= c7;
            if (v8) begin
                shifted_q  <= shifted_nx;
                sat_q      <= sat_nx;
                sat_sign_q <= sat_sign_nx;
            end
        end
    end

    // ---------------------------------------------------------------- S10: row accumulation
    reg signed [63:0] row_acc_64;
    reg               sat_sticky;   // row accumulator has saturated (acts like +inf)
    reg               sat_sign;     // saturated sign: 0 = +inf, 1 = -inf

    always @(posedge aclk) begin
        if (!rst_n) begin
            v9         <= 1'b0;
            c9         <= 11'd0;
            row_acc_64 <= ACC_BIAS;
            sat_sticky <= 1'b0;
            sat_sign   <= 1'b0;
        end else begin
            v9 <= v8;
            c9 <= c8;
            // Consecutive block-end stages are >= BEATS_PER_BLOCK cycles apart, so the
            // accumulate branch (block n) and the row-reset branch (block n-1) can never
            // fire on the same edge.  The reset is written last purely as a safety net.
            if (v9 && dequant_en) begin
                if (sat_sticky)
                    row_acc_64 <= row_acc_64;                        // infinity absorbs
                else if (sat_q)
                    row_acc_64 <= sat_sign_q ? ACC_MIN64 : ACC_MAX64;
                else
                    row_acc_64 <= row_acc_64 + shifted_q;
                if (!sat_sticky && sat_q) begin
                    sat_sticky <= 1'b1;
                    sat_sign   <= sat_sign_q;
                end
            end
            if (v10 && dequant_en && c10[9]) begin
                row_acc_64 <= ACC_BIAS;
                sat_sticky <= 1'b0;
                sat_sign   <= 1'b0;
            end
        end
    end

    wire       c10_gemv_last = c10[10];
    wire       c10_row_last  = c10[9];

    always @(posedge aclk) begin
        if (!rst_n) begin
            v10 <= 1'b0;
            c10 <= 11'd0;
        end else begin
            v10 <= v9;
            c10 <= c9;
        end
    end

    // ---------------------------------------------------------------- output beat / FIFO
    // dequant_en = 1 : one beat per row, pushed at S10 from the finished row accumulator.
    //                  The conversion is a pure bit slice (bias already folded in);
    //                  a saturated row clamps to the int32 extremes.
    // dequant_en = 0 : one beat per block, pushed at S6 straight from bd_q (legacy).
    wire [31:0] of_data_nx = dequant_en
                           ? (sat_sticky ? (sat_sign ? 32'sh8000_0000 : 32'sh7FFF_FFFF)
                                         : row_acc_64[63:20])
                           : bd_q;
    wire        of_last_nx = dequant_en ? c10_gemv_last : c6_gemv_last;
    wire        of_push    = dequant_en ? (v10 && c10_row_last) : (v6 && !dequant_en);

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
        end else begin
            if (of_push) begin
                of_data[of_wp] <= of_data_nx;
                of_last[of_wp] <= of_last_nx;
                of_wp          <= of_wp + 3'd1;
            end
            if (of_pop) of_rp <= of_rp + 3'd1;

            of_cnt        <= of_cnt_next;
            m_axis_tvalid <= (of_cnt_next != 4'd0);
        end
    end

    // ---------------------------------------------------------------- main FSM
    integer k;
    always @(posedge aclk) begin
        if (!rst_n) begin
            state        <= ST_HDR;
            x_last       <= 16'd0;
            bpr_m1       <= 8'd0;
            dequant_en   <= 1'b0;
            exp_beats_m1 <= 7'd0;
            exp_cnt      <= 7'd0;
            x_load_cnt   <= 16'd0;
            x_hold       <= 128'd0;
            xw_we        <= 1'b0;
            xw_waddr     <= {XW{1'b0}};
            xw_wdata     <= 256'd0;
            pos          <= 8'd0;
            row_toggle   <= 1'b0;
            for (k = 0; k < 128; k = k + 1) begin
                row_exp[0][k] <= 8'd134;
                row_exp[1][k] <= 8'd134;
            end
        end else begin
            xw_we <= 1'b0;

            case (state)
            ST_HDR: begin
                if (s_fire) begin
                    x_last       <= s_axis_tdata[15:0] - 16'd1;
                    bpr_m1       <= s_axis_tdata[23:16];
                    exp_beats_m1 <= s_axis_tdata[30:24];
                    dequant_en   <= s_axis_tdata[31];
                    exp_cnt      <= 7'd0;
                    x_load_cnt   <= 16'd0;
                    pos          <= 8'd0;
                    row_toggle   <= 1'b0;
                    if (s_axis_tdata[15:0] != 16'd0) begin
                        state <= ST_LOADX;
                    end else if (s_axis_tdata[31]) begin
                        state <= ST_EXP;
                    end else begin
                        state <= ST_RUN;
                    end
                end
            end

            ST_LOADX: begin
                if (s_fire) begin
                    if (x_load_cnt[0] == 1'b0) begin
                        x_hold <= s_axis_tdata;
                    end else begin
                        xw_we    <= 1'b1;
                        xw_waddr <= x_load_cnt[XW:1];
                        xw_wdata <= {s_axis_tdata, x_hold};
                    end
                    x_load_cnt <= x_load_cnt + 16'd1;
                    if (x_load_cnt == x_last) begin
                        state   <= dequant_en ? ST_EXP : ST_RUN;
                        exp_cnt <= 7'd0;
                        pos     <= 8'd0;
                    end
                end
            end

            ST_EXP: begin
                if (s_fire) begin
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

            ST_RUN: begin
                if (s_fire) begin
                    pos <= (pos == bpr_m1) ? 8'd0 : (pos + 8'd1);
                    if (s_axis_tlast) begin
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
                // Wait for the whole pipeline (all ten stages) to retire before a new
                // header can be accepted, so dequant_en / bpr_m1 cannot change under a
                // beat that is still in flight.
                if (!v1 && !v2 && !v3 && !v4 && !v5 &&
                    !v6 && !v7 && !v8 && !v9 && !v10) state <= ST_HDR;
            end

            default: state <= ST_HDR;
            endcase
        end
    end

    // ---------------------------------------------------------------- debug
    assign dbg = {m_axis_tlast, m_axis_tready, m_axis_tvalid,
                  s_axis_tready, s_axis_tvalid, state,
                  pos, x_load_cnt};

endmodule
