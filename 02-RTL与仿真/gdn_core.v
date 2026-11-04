//=============================================================================
// gdn_core.v  --  Gated DeltaNet in-place state recurrence accelerator
//
// Target : Xilinx Zynq-7020 (XC7Z020-2CLG400I), 166.67 MHz (6.0 ns)
// Style  : Verilog-2001, single clock domain (aclk), async active-low reset
//
//---------------------------------------------------------------------------
// STATE STORAGE / BANK LAYOUT
//---------------------------------------------------------------------------
// 8 heads x 64 rows x 64 cols x 16 bit = 64 KB, split into 16 parallel banks
// so that the 16 MAC lanes each hit a different bank every cycle.
//
//   element (h, i, j)  ->  bank  = j % 16        (== lane index)
//                          word  = (h*64 + i)*4 + (j/16)
//
// Each bank is its own flat 1-D array of 2048 words of 16 bit (bank0..bank15,
// declared individually so that BRAM inference is unambiguous).  A 64-element
// row is 4 words per lane, i.e. 4 lane-groups of 16 parallel MACs per row.
//
// Both passes over the state use the same single port:
//   pass 1 (phase 2) : u[j] += k[i] * S[i][j]          read only
//   pass 2 (phase 4) : S[i][j] = alpha*S + beta*k*e    read + write back
//                      o[j]   += q[i] * S_new
//
//---------------------------------------------------------------------------
// TIMING / PIPELINING
//---------------------------------------------------------------------------
// The 6.0 ns period allows roughly one DSP48E1 plus one short carry chain per
// register stage, and NOT a BRAM embedded-output-register clock-to-out
// (2.1 ns) followed by a DSP48E1 (3.1 ns).  The state read is therefore two
// cycles deep: address -> BRAM DOADO (rd_dataN) -> lane_q.  Every multiplier
// starts from a flip-flop.
//
// Phase 4 is cut into four pipeline stages per lane-group (p0..p3), one group
// per 4 cycles, so a row takes 16 cycles and the phase-4 pass 1024 cycles:
//
//   p0 : present bank address for group g; form prod = q*S_new of group g-1;
//        accumulate o[g-2] += prod; latch k[i]
//   p1 : form ke = k[i]*e[j]; latch q[i]
//   p2 : S_old(g) has reached lane_q; form aS = (alpha*S_old)>>16 and
//        bke = (beta*ke)>>31
//   p3 : upd = aS + bke, saturate, write S_new back, latch it for p0
//
// ke/bke do not depend on the state read, which is what lets the extra read
// stage fit without an extra cycle.  Every multiplier starts from a flip-flop,
// and no register stage holds a cascaded-DSP pair: the 17x32 beta*ke product
// is split into two PARALLEL 17x17 DSP48E1 multiplies (see g_bke), whose
// results are combined one stage later.  p2 is therefore two independent
// single-DSP multiplies, and p3 is one short carry chain plus the saturate.
//
// Phase 2 keeps its 2-cycle-per-group shape (product cycle then accumulate
// cycle): a row is 8 cycles and the pass 512 cycles.  Because the read is two
// cycles deep, group g of a row is latched at the start of group g+1 (its data
// has arrived by then) and added one cycle later; the last group of the last
// row is picked up by the two-cycle ST_P2D drain.  The accumulate order per
// lane is unchanged.
//
// A token is therefore ~1671 cycles: 64 input beats + 514 (phase 2 incl. drain)
// + 2 (phase 3) + 1026 (phase 4 incl. drain) + 1 (output quantisation) + 64
// output beats.
//
//---------------------------------------------------------------------------
// FIXED POINT
//---------------------------------------------------------------------------
//   q,k,v : Q2.14 signed 16      alpha,beta : Q0.16 UNSIGNED 16
//   S     : Q4.12 signed 16      o          : Q2.14 signed 16
//   u     : Q6.26 in 32-bit acc  e          : Q3.13 signed 16
// All rounding/saturation funnels through sat16(v, shift) (round half up,
// then saturate to the signed 16-bit range).
//
// alpha and beta are unsigned Q0.16 values in [0,1); they are zero-extended
// to 17-bit signed before multiplication so that codes >= 0x8000 (>= 0.5) are
// treated as positive.
//
//---------------------------------------------------------------------------
// MODE HANDLING
//---------------------------------------------------------------------------
// mode==0 (decode) is level sensitive and free running: after the 64 output
// beats the core returns to IDLE and immediately re-arms for the next token.
// mode==1 (reset) and mode==2 (dump) are re-triggered only when the mode input
// changes value, so a mode held high does not loop forever.
//=============================================================================
`timescale 1ns / 1ps

module gdn_core (
    input  wire        aclk,
    input  wire        aresetn,
    input  wire [63:0] s_axis_tdata,   // {last, v[15:0], k[15:0], q[15:0]}
    input  wire        s_axis_tvalid,
    output wire        s_axis_tready,
    input  wire        s_axis_tlast,
    output wire [31:0] m_axis_tdata,   // {last, value[15:0]}
    output wire        m_axis_tvalid,
    input  wire        m_axis_tready,
    output wire        m_axis_tlast,
    input  wire [7:0]  mode,
    input  wire [7:0]  head_idx,
    input  wire [15:0] alpha,
    input  wire [15:0] beta
);

    //-------------------------------------------------------------------------
    // sat16 : round-to-nearest (half up) right shift, then saturate to signed
    // 16-bit.  shift > 0 : round + arithmetic shift right
    //           shift = 0 : saturate only
    //           shift < 0 : left shift then saturate
    //-------------------------------------------------------------------------
    function signed [15:0] sat16;
        input signed [31:0] v;
        input integer       shift;
        reg   signed [32:0] t;
        begin
            t = {v[31], v};
            if (shift > 0) begin
                t = (t + (33'sd1 <<< (shift - 1))) >>> shift;
            end else if (shift < 0) begin
                t = t <<< (-shift);
            end
            if (t > 33'sd32767)      sat16 = 16'sh7FFF;
            else if (t < -33'sd32768) sat16 = 16'sh8000;
            else                     sat16 = t[15:0];
        end
    endfunction

    //-------------------------------------------------------------------------
    // Constants
    //-------------------------------------------------------------------------
    localparam [7:0] MODE_DECODE = 8'd0;
    localparam [7:0] MODE_RESET  = 8'd1;
    localparam [7:0] MODE_DUMP   = 8'd2;

    localparam [3:0] ST_IDLE = 4'd0;
    localparam [3:0] ST_RST  = 4'd1;
    localparam [3:0] ST_DUMP = 4'd2;
    localparam [3:0] ST_IN   = 4'd3;
    localparam [3:0] ST_P2   = 4'd4;
    localparam [3:0] ST_P3   = 4'd5;
    localparam [3:0] ST_P4   = 4'd6;
    localparam [3:0] ST_OUT  = 4'd7;
    localparam [3:0] ST_P4D  = 4'd8;
    localparam [3:0] ST_OQ   = 4'd9;
    localparam [3:0] ST_P2D  = 4'd10;

    //-------------------------------------------------------------------------
    // State memory : 16 banks, 2048 x 16 each (no reset, BRAM is zero at config)
    //-------------------------------------------------------------------------
    (* ram_style = "block" *) reg [15:0] bank0  [0:2047];
    (* ram_style = "block" *) reg [15:0] bank1  [0:2047];
    (* ram_style = "block" *) reg [15:0] bank2  [0:2047];
    (* ram_style = "block" *) reg [15:0] bank3  [0:2047];
    (* ram_style = "block" *) reg [15:0] bank4  [0:2047];
    (* ram_style = "block" *) reg [15:0] bank5  [0:2047];
    (* ram_style = "block" *) reg [15:0] bank6  [0:2047];
    (* ram_style = "block" *) reg [15:0] bank7  [0:2047];
    (* ram_style = "block" *) reg [15:0] bank8  [0:2047];
    (* ram_style = "block" *) reg [15:0] bank9  [0:2047];
    (* ram_style = "block" *) reg [15:0] bank10 [0:2047];
    (* ram_style = "block" *) reg [15:0] bank11 [0:2047];
    (* ram_style = "block" *) reg [15:0] bank12 [0:2047];
    (* ram_style = "block" *) reg [15:0] bank13 [0:2047];
    (* ram_style = "block" *) reg [15:0] bank14 [0:2047];
    (* ram_style = "block" *) reg [15:0] bank15 [0:2047];

    //-------------------------------------------------------------------------
    // Sample buffers and per-token vectors
    //-------------------------------------------------------------------------
    reg signed [15:0] q_buf [0:63];   // Q2.14
    reg signed [15:0] k_buf [0:63];   // Q2.14
    reg signed [15:0] v_buf [0:63];   // Q2.14
    reg signed [15:0] u_q   [0:63];   // Q2.14, quantised u
    reg signed [15:0] e_buf [0:63];   // Q3.13, innovation

    reg signed [31:0] u_mac [0:3][0:15];  // Q6.26 running accumulator
    reg signed [31:0] o_mac [0:3][0:15];  // Q6.26 running accumulator

    //-------------------------------------------------------------------------
    // Phase-4 pipeline registers (one entry per lane)
    //-------------------------------------------------------------------------
    reg signed [31:0] ke_r   [0:15];  // k*e[j],            Q5.27
    reg signed [16:0] aS_r   [0:15];  // (alpha*S_old)>>16,  Q4.12
    reg signed [31:0] bke_hi_r [0:15];  // beta*ke[31:16],  Q5.27 high half
    reg signed [16:0] bke_lo_r [0:15];  // (beta*ke[15:0])>>16
    reg signed [15:0] snew_r [0:15];  // S_new,              Q4.12
    reg signed [31:0] prod_r [0:15];  // q*S_new,            Q6.26
    reg [1:0] prod_grp;
    reg       prod_vld;

    //-------------------------------------------------------------------------
    // Phase-2 product / accumulate split
    //   p0 latches the product of group g-1 (state data on lane_q); p1 adds it
    //   to u_mac.  Keeping the DSP48E1 output and the 32-bit accumulate in
    //   different cycles removes the u_mac feedback mux from the DSP path.
    //-------------------------------------------------------------------------
    reg signed [31:0] p2_prod_r [0:15];
    reg [1:0]         p2_grp_r;
    reg               p2_acc_vld;

    //-------------------------------------------------------------------------
    // Control
    //-------------------------------------------------------------------------
    reg [3:0]  state;
    reg [3:0]  sub;          // P2: 0..7 (group/phase), P4: 0..15 (group/phase)
    reg [5:0]  row_cnt;
    reg [5:0]  out_cnt;
    reg [5:0]  ld_cnt;
    reg [10:0] rst_cnt;
    reg [12:0] dump_cnt;
    reg [5:0]  dump_jd;
    reg [12:0] dump_id;
    reg        dump_issue;

    reg [7:0]  head_sel;
    reg [7:0]  mode_hold;
    reg [15:0] alpha_u;      // Q0.16 unsigned
    reg [15:0] beta_u;       // Q0.16 unsigned

    reg signed [15:0] k_cur;
    reg signed [15:0] q_cur;

    reg        tready_r;
    reg        tvalid_r;
    reg [31:0] tdata_r;
    reg        tlast_r;

    integer g;
    integer l;

    //-------------------------------------------------------------------------
    // Bank port addressing (shared by the two state passes, reset and dump)
    //-------------------------------------------------------------------------
    wire [10:0] base_addr = {head_sel[2:0], row_cnt[5:0], 2'b00};  // (h*64+i)*4
    wire [1:0]  grp_sel   = (state == ST_P4) ? sub[3:2] : sub[2:1];
    wire [10:0] grp_addr  = base_addr + {9'd0, grp_sel};
    wire [10:0] dump_addr = {head_sel[2:0], dump_cnt[11:6], 2'b00}
                            + {9'd0, dump_cnt[5:4]};
    wire [10:0] rst_base  = (head_sel < 8'd8) ? {head_sel[2:0], 8'd0} : 11'd0;
    wire [10:0] rst_last  = (head_sel < 8'd8) ? 11'd255 : 11'd2047;
    wire [10:0] rst_addr  = rst_base + rst_cnt;

    wire [10:0] bank_addr = (state == ST_DUMP) ? dump_addr :
                            (state == ST_RST)  ? rst_addr  : grp_addr;

    wire bank_we = (state == ST_RST) ||
                   ((state == ST_P4) && (sub[1:0] == 2'd3));

    //-------------------------------------------------------------------------
    // Registered bank read data (one lane per bank)
    //-------------------------------------------------------------------------
    reg [15:0] rd_data0,  rd_data1,  rd_data2,  rd_data3;
    reg [15:0] rd_data4,  rd_data5,  rd_data6,  rd_data7;
    reg [15:0] rd_data8,  rd_data9,  rd_data10, rd_data11;
    reg [15:0] rd_data12, rd_data13, rd_data14, rd_data15;

    wire [15:0] lane_rd [0:15];
    assign lane_rd[0]  = rd_data0;
    assign lane_rd[1]  = rd_data1;
    assign lane_rd[2]  = rd_data2;
    assign lane_rd[3]  = rd_data3;
    assign lane_rd[4]  = rd_data4;
    assign lane_rd[5]  = rd_data5;
    assign lane_rd[6]  = rd_data6;
    assign lane_rd[7]  = rd_data7;
    assign lane_rd[8]  = rd_data8;
    assign lane_rd[9]  = rd_data9;
    assign lane_rd[10] = rd_data10;
    assign lane_rd[11] = rd_data11;
    assign lane_rd[12] = rd_data12;
    assign lane_rd[13] = rd_data13;
    assign lane_rd[14] = rd_data14;
    assign lane_rd[15] = rd_data15;

    //-------------------------------------------------------------------------
    // Second read stage.  lane_rd is the BRAM embedded output register, whose
    // clock-to-out (2.1 ns) plus a DSP48E1 (3.1 ns) plus routing does not fit
    // in 6.0 ns.  lane_q is one more fabric register, so every state read is
    // available two cycles after its address is presented and the multiplier
    // always starts from a flip-flop.  Only the dump port still uses lane_rd.
    //-------------------------------------------------------------------------
    reg [15:0] lane_q [0:15];

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            lane_q[0]  <= 16'd0;
            lane_q[1]  <= 16'd0;
            lane_q[2]  <= 16'd0;
            lane_q[3]  <= 16'd0;
            lane_q[4]  <= 16'd0;
            lane_q[5]  <= 16'd0;
            lane_q[6]  <= 16'd0;
            lane_q[7]  <= 16'd0;
            lane_q[8]  <= 16'd0;
            lane_q[9]  <= 16'd0;
            lane_q[10] <= 16'd0;
            lane_q[11] <= 16'd0;
            lane_q[12] <= 16'd0;
            lane_q[13] <= 16'd0;
            lane_q[14] <= 16'd0;
            lane_q[15] <= 16'd0;
        end else begin
            lane_q[0]  <= rd_data0;
            lane_q[1]  <= rd_data1;
            lane_q[2]  <= rd_data2;
            lane_q[3]  <= rd_data3;
            lane_q[4]  <= rd_data4;
            lane_q[5]  <= rd_data5;
            lane_q[6]  <= rd_data6;
            lane_q[7]  <= rd_data7;
            lane_q[8]  <= rd_data8;
            lane_q[9]  <= rd_data9;
            lane_q[10] <= rd_data10;
            lane_q[11] <= rd_data11;
            lane_q[12] <= rd_data12;
            lane_q[13] <= rd_data13;
            lane_q[14] <= rd_data14;
            lane_q[15] <= rd_data15;
        end
    end

    //-------------------------------------------------------------------------
    // Per-lane datapath
    //   mul_a/mul_s : one shared 16x16 multiplier per lane
    //                 phase 2 -> k[i] * S_old      (Q6.26) -> p2_prod
    //                 phase 4 -> alpha * S_old     (Q4.28) -> aS
    //                 both read lane_q (2-cycle-delayed read data)
    //   ke          : k[i] * e[j]                   (Q5.27)
    //   bke         : (beta * ke) >> 31             (Q4.12)
    //   upd         : aS + bke                      (Q4.12)
    //   prod        : q[i] * S_new                  (Q6.26)
    //-------------------------------------------------------------------------
    wire signed [16:0] mul_a_nxt = ((state == ST_P2) || (state == ST_P2D))
                                   ? $signed(k_cur)
                                   : $signed({1'b0, alpha_u});

    // Registered operand select: keeps `state` (a high-fanout control bit) off
    // the DSP48E1 A port, which was worth >1 ns of routing in the routed
    // out-of-context report.  mul_a_r is always the value mul_a_nxt had in the
    // previous cycle; in phase 2 that is still the k[i] of the row being
    // accumulated (k_cur only changes at p0), and in phase 4 it is alpha_u.
    reg signed [16:0] mul_a_r;
    wire signed [31:0] p2_prod [0:15];
    wire signed [31:0] mul_s   [0:15];
    wire signed [16:0] aS_w    [0:15];
    wire signed [31:0] ke_w    [0:15];
    wire signed [31:0] bke_hi_w[0:15];
    wire signed [16:0] bke_lo_w[0:15];
    wire signed [16:0] bke_w   [0:15];
    wire signed [31:0] upd_w   [0:15];
    wire signed [15:0] p4_snew [0:15];
    wire signed [31:0] prod_w  [0:15];

    genvar gi;
    generate
        for (gi = 0; gi < 16; gi = gi + 1) begin : g_lane
            localparam [3:0] LID = gi;
            wire [5:0] j_idx = {sub[3:2], LID};   // phase-4 column index

            // one shared 16x16 multiplier (k*S in P2, alpha*S in P4)
            assign mul_s[gi] = $signed(mul_a_r) * $signed(lane_q[gi]);
            assign p2_prod[gi] = mul_s[gi];
            assign aS_w[gi]   = mul_s[gi] >>> 16;

            // k[i] * e[j]   (Q2.14 * Q3.13 -> Q5.27)
            assign ke_w[gi] = $signed(k_cur) * $signed(e_buf[j_idx]);

            // (beta * ke) >> 31, split so that neither multiply carries a
            // 32-bit operand.  A 17x32 product costs two CASCADED DSP48E1
            // (about 4.7 ns of logic plus the cascade route) and was the
            // critical path; two 17x17 products run in PARALLEL instead.
            //
            //   ke    = ke_hi*2^16 + ke_lo        (ke_lo unsigned)
            //   beta*ke = beta*ke_hi*2^16 + beta*ke_lo
            //
            // so with D = beta*ke_hi and C = beta*ke_lo >= 0
            //
            //   (beta*ke) >> 31 = (D + (C >> 16)) >>> 15
            //
            // exactly: (D*2^16 + C)/2^31 = (D + C/2^16)/2^15, and the
            // fractional part f = (C mod 2^16)/2^16 in [0,1) cannot change the
            // floor of (D + C/2^16)/2^15.  The two products are latched and
            // combined in the next stage (see bke_w below), which also stops
            // the tool from folding the add back into a DSP cascade.
            begin : g_bke
                wire signed [16:0] beta_x = {1'b0, beta_u};
                wire signed [16:0] ke_hi  = {{1{ke_r[gi][31]}}, ke_r[gi][31:16]};
                wire signed [16:0] ke_lo  = {1'b0, ke_r[gi][15:0]};
                wire signed [33:0] p_hi   = beta_x * ke_hi;
                wire signed [33:0] p_lo   = beta_x * ke_lo;
                assign bke_hi_w[gi] = p_hi[31:0];     // |.| <= 1.074e9
                assign bke_lo_w[gi] = p_lo[32:16];    // 0 .. 65535
            end

            // bke = (D + (C >> 16)) >>> 15, then aS + bke, then saturate
            assign bke_w[gi]   = ({{1{bke_hi_r[gi][31]}}, bke_hi_r[gi]}
                                  + {16'd0, bke_lo_r[gi]}) >>> 15;
            assign upd_w[gi]   = {{15{aS_r[gi][16]}}, aS_r[gi]}
                               + {{15{bke_w[gi][16]}}, bke_w[gi]};
            assign p4_snew[gi] = sat16(upd_w[gi], 0);

            // q[i] * S_new   (Q2.14 * Q4.12 -> Q6.26)
            assign prod_w[gi] = $signed(q_cur) * $signed(snew_r[gi]);
        end
    endgenerate

    //-------------------------------------------------------------------------
    // Output quantisation, Q6.26 -> Q2.14
    //   The saturating shift is registered (ST_OQ) so that the 64:1 output
    //   multiplexer in ST_OUT only sees register outputs: the previous form
    //   put 64 sat16 adders in series with the 64:1 mux and was the critical
    //   path.  The arithmetic is unchanged.
    //-------------------------------------------------------------------------
    reg signed [15:0] o_q_r [0:63];

    //-------------------------------------------------------------------------
    // Streaming ports
    //-------------------------------------------------------------------------
    assign s_axis_tready = tready_r;
    assign m_axis_tvalid = tvalid_r;
    assign m_axis_tdata  = tdata_r;
    assign m_axis_tlast  = tlast_r;

    //-------------------------------------------------------------------------
    // Banks : one synchronous port each (write + registered read, read-first)
    //-------------------------------------------------------------------------
    always @(posedge aclk) begin
        if (bank_we) bank0[bank_addr] <= (state == ST_RST) ? 16'd0 : p4_snew[0];
        rd_data0 <= bank0[bank_addr];
    end
    always @(posedge aclk) begin
        if (bank_we) bank1[bank_addr] <= (state == ST_RST) ? 16'd0 : p4_snew[1];
        rd_data1 <= bank1[bank_addr];
    end
    always @(posedge aclk) begin
        if (bank_we) bank2[bank_addr] <= (state == ST_RST) ? 16'd0 : p4_snew[2];
        rd_data2 <= bank2[bank_addr];
    end
    always @(posedge aclk) begin
        if (bank_we) bank3[bank_addr] <= (state == ST_RST) ? 16'd0 : p4_snew[3];
        rd_data3 <= bank3[bank_addr];
    end
    always @(posedge aclk) begin
        if (bank_we) bank4[bank_addr] <= (state == ST_RST) ? 16'd0 : p4_snew[4];
        rd_data4 <= bank4[bank_addr];
    end
    always @(posedge aclk) begin
        if (bank_we) bank5[bank_addr] <= (state == ST_RST) ? 16'd0 : p4_snew[5];
        rd_data5 <= bank5[bank_addr];
    end
    always @(posedge aclk) begin
        if (bank_we) bank6[bank_addr] <= (state == ST_RST) ? 16'd0 : p4_snew[6];
        rd_data6 <= bank6[bank_addr];
    end
    always @(posedge aclk) begin
        if (bank_we) bank7[bank_addr] <= (state == ST_RST) ? 16'd0 : p4_snew[7];
        rd_data7 <= bank7[bank_addr];
    end
    always @(posedge aclk) begin
        if (bank_we) bank8[bank_addr] <= (state == ST_RST) ? 16'd0 : p4_snew[8];
        rd_data8 <= bank8[bank_addr];
    end
    always @(posedge aclk) begin
        if (bank_we) bank9[bank_addr] <= (state == ST_RST) ? 16'd0 : p4_snew[9];
        rd_data9 <= bank9[bank_addr];
    end
    always @(posedge aclk) begin
        if (bank_we) bank10[bank_addr] <= (state == ST_RST) ? 16'd0 : p4_snew[10];
        rd_data10 <= bank10[bank_addr];
    end
    always @(posedge aclk) begin
        if (bank_we) bank11[bank_addr] <= (state == ST_RST) ? 16'd0 : p4_snew[11];
        rd_data11 <= bank11[bank_addr];
    end
    always @(posedge aclk) begin
        if (bank_we) bank12[bank_addr] <= (state == ST_RST) ? 16'd0 : p4_snew[12];
        rd_data12 <= bank12[bank_addr];
    end
    always @(posedge aclk) begin
        if (bank_we) bank13[bank_addr] <= (state == ST_RST) ? 16'd0 : p4_snew[13];
        rd_data13 <= bank13[bank_addr];
    end
    always @(posedge aclk) begin
        if (bank_we) bank14[bank_addr] <= (state == ST_RST) ? 16'd0 : p4_snew[14];
        rd_data14 <= bank14[bank_addr];
    end
    always @(posedge aclk) begin
        if (bank_we) bank15[bank_addr] <= (state == ST_RST) ? 16'd0 : p4_snew[15];
        rd_data15 <= bank15[bank_addr];
    end

    //-------------------------------------------------------------------------
    // Main sequential process
    //-------------------------------------------------------------------------
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            state      <= ST_IDLE;
            sub        <= 4'd0;
            row_cnt    <= 6'd0;
            out_cnt    <= 6'd0;
            ld_cnt     <= 6'd0;
            rst_cnt    <= 11'd0;
            dump_cnt   <= 13'd0;
            dump_jd    <= 6'd0;
            dump_id    <= 13'd0;
            dump_issue <= 1'b0;
            head_sel   <= 8'd0;
            mode_hold  <= 8'hFF;
            alpha_u    <= 16'd0;
            beta_u     <= 16'd0;
            k_cur      <= 16'sd0;
            q_cur      <= 16'sd0;
            prod_grp   <= 2'd0;
            prod_vld   <= 1'b0;
            p2_grp_r   <= 2'd0;
            p2_acc_vld <= 1'b0;
            mul_a_r    <= 17'sd0;
            tready_r   <= 1'b0;
            tvalid_r   <= 1'b0;
            tdata_r    <= 32'd0;
            tlast_r    <= 1'b0;
            for (g = 0; g < 4; g = g + 1) begin
                for (l = 0; l < 16; l = l + 1) begin
                    u_mac[g][l] <= 32'sd0;
                    o_mac[g][l] <= 32'sd0;
                end
            end
            for (g = 0; g < 16; g = g + 1) begin
                ke_r[g]      <= 32'sd0;
                aS_r[g]      <= 17'sd0;
                bke_hi_r[g]  <= 32'sd0;
                bke_lo_r[g]  <= 17'sd0;
                snew_r[g]    <= 16'sd0;
                prod_r[g]    <= 32'sd0;
                p2_prod_r[g] <= 32'sd0;
            end
            for (g = 0; g < 64; g = g + 1) begin
                q_buf[g] <= 16'sd0;
                k_buf[g] <= 16'sd0;
                v_buf[g] <= 16'sd0;
                u_q[g]   <= 16'sd0;
                e_buf[g] <= 16'sd0;
                o_q_r[g] <= 16'sd0;
            end
        end else begin
            mul_a_r <= mul_a_nxt;
            case (state)

            //-----------------------------------------------------------------
            ST_IDLE: begin
                tready_r <= 1'b0;
                if (tvalid_r) begin
                    // drain the final output beat before accepting a new op
                    if (m_axis_tready) begin
                        tvalid_r <= 1'b0;
                        tlast_r  <= 1'b0;
                    end
                end else begin
                    tlast_r <= 1'b0;
                    if (mode == MODE_RESET && mode != mode_hold) begin
                        mode_hold <= mode;
                        head_sel  <= head_idx;
                        rst_cnt   <= 11'd0;
                        state     <= ST_RST;
                    end else if (mode == MODE_DUMP && mode != mode_hold) begin
                        mode_hold  <= mode;
                        head_sel   <= (head_idx < 8'd8) ? head_idx : 8'd0;
                        dump_cnt   <= 13'd0;
                        dump_issue <= 1'b0;
                        state      <= ST_DUMP;
                    end else if (mode == MODE_DECODE) begin
                        mode_hold <= 8'hFF;
                        head_sel  <= (head_idx < 8'd8) ? head_idx : 8'd0;
                        alpha_u   <= alpha;
                        beta_u    <= beta;
                        ld_cnt    <= 6'd0;
                        tready_r  <= 1'b1;
                        state     <= ST_IN;
                    end
                end
            end

            //-----------------------------------------------------------------
            // MODE 1 : zero the state matrix
            //-----------------------------------------------------------------
            ST_RST: begin
                tready_r <= 1'b1;
                if (rst_cnt == rst_last) begin
                    rst_cnt  <= 11'd0;
                    tready_r <= 1'b0;
                    state    <= ST_IDLE;
                end else begin
                    rst_cnt <= rst_cnt + 11'd1;
                end
            end

            //-----------------------------------------------------------------
            // MODE 2 : dump 4096 beats, row-major (i outer, j inner)
            //-----------------------------------------------------------------
            ST_DUMP: begin
                tready_r <= 1'b0;
                if (tvalid_r && !m_axis_tready) begin
                    // output register busy, hold everything
                end else if (dump_issue) begin
                    tdata_r    <= {15'd0, (dump_id == 13'd4095), lane_rd[dump_jd[3:0]]};
                    tlast_r    <= (dump_id == 13'd4095);
                    tvalid_r   <= 1'b1;
                    dump_issue <= 1'b0;
                end else if (dump_cnt <= 13'd4095) begin
                    dump_jd    <= dump_cnt[5:0];
                    dump_id    <= dump_cnt;
                    dump_issue <= 1'b1;
                    dump_cnt   <= dump_cnt + 13'd1;
                    tvalid_r   <= 1'b0;
                end else begin
                    tvalid_r  <= 1'b0;
                    tlast_r   <= 1'b0;
                    dump_cnt  <= 13'd0;
                    state     <= ST_IDLE;
                end
            end

            //-----------------------------------------------------------------
            // Phase 1 : capture 64 input packets
            //-----------------------------------------------------------------
            ST_IN: begin
                tready_r <= 1'b1;
                if (s_axis_tvalid && tready_r) begin
                    q_buf[ld_cnt] <= s_axis_tdata[15:0];
                    k_buf[ld_cnt] <= s_axis_tdata[31:16];
                    v_buf[ld_cnt] <= s_axis_tdata[47:32];
                    if (ld_cnt == 6'd63) begin
                        tready_r <= 1'b0;
                        ld_cnt   <= 6'd0;
                        row_cnt  <= 6'd0;
                        sub      <= 4'd0;
                        for (g = 0; g < 4; g = g + 1) begin
                            for (l = 0; l < 16; l = l + 1) begin
                                u_mac[g][l] <= 32'sd0;
                            end
                        end
                        state <= ST_P2;
                    end else begin
                        ld_cnt <= ld_cnt + 6'd1;
                    end
                end
            end

            //-----------------------------------------------------------------
            //-----------------------------------------------------------------
            // Phase 2 : u = k^T * S, one row per 8 cycles
            //   The state read is two cycles deep (address -> BRAM DOADO ->
            //   lane_q), so group g of a row is latched at the start of group
            //   g+1, when its data has arrived, and added to u_mac one cycle
            //   later.  The first group has no predecessor and the last group
            //   of the last row is picked up by the ST_P2D drain.
            //-----------------------------------------------------------------
            ST_P2: begin
                if (sub[0] == 1'b0) begin
                    // p0 : present the address of group sub[2:1] and latch the
                    //      product of group sub[2:1]-1, whose data is on lane_q
                    k_cur      <= k_buf[row_cnt];
                    p2_grp_r   <= sub[2:1] - 2'd1;
                    p2_acc_vld <= !((sub == 4'd0) && (row_cnt == 6'd0));
                    for (l = 0; l < 16; l = l + 1) begin
                        p2_prod_r[l] <= p2_prod[l];
                    end
                end else begin
                    // p1 : accumulate the latched product
                    if (p2_acc_vld) begin
                        for (l = 0; l < 16; l = l + 1) begin
                            u_mac[p2_grp_r][l] <= u_mac[p2_grp_r][l] + p2_prod_r[l];
                        end
                    end
                end
                if (sub == 4'd7) begin
                    sub <= 4'd0;
                    if (row_cnt == 6'd63) begin
                        state <= ST_P2D;
                    end else begin
                        row_cnt <= row_cnt + 6'd1;
                    end
                end else begin
                    sub <= sub + 4'd1;
                end
            end

            //-----------------------------------------------------------------
            // Phase 2 drain : latch and accumulate the last group
            //                 (group 3 of row 63)
            //-----------------------------------------------------------------
            ST_P2D: begin
                if (sub == 4'd0) begin
                    p2_grp_r   <= 2'd3;
                    p2_acc_vld <= 1'b1;
                    for (l = 0; l < 16; l = l + 1) begin
                        p2_prod_r[l] <= p2_prod[l];
                    end
                    sub <= 4'd1;
                end else begin
                    for (l = 0; l < 16; l = l + 1) begin
                        u_mac[2'd3][l] <= u_mac[2'd3][l] + p2_prod_r[l];
                    end
                    p2_acc_vld <= 1'b0;
                    row_cnt    <= 6'd0;
                    sub        <= 4'd0;
                    state      <= ST_P3;
                end
            end

            //-----------------------------------------------------------------
            // Phase 3 : u_quant then e[j] = v[j] - u_quant
            //-----------------------------------------------------------------
            ST_P3: begin
                if (sub == 4'd0) begin
                    for (g = 0; g < 4; g = g + 1) begin
                        for (l = 0; l < 16; l = l + 1) begin
                            u_q[g*16+l] <= sat16(u_mac[g][l], 12);
                        end
                    end
                    sub <= 4'd1;
                end else begin
                    for (g = 0; g < 4; g = g + 1) begin
                        for (l = 0; l < 16; l = l + 1) begin
                            e_buf[g*16+l] <=
                                sat16({{16{v_buf[g*16+l][15]}}, v_buf[g*16+l]} -
                                      {{16{u_q[g*16+l][15]}},   u_q[g*16+l]}, 0);
                            o_mac[g][l] <= 32'sd0;
                        end
                    end
                    sub      <= 4'd0;
                    row_cnt  <= 6'd0;
                    prod_vld <= 1'b0;
                    prod_grp <= 2'd0;
                    state    <= ST_P4;
                end
            end

            //-----------------------------------------------------------------
            // Phase 4 : in-place update + o accumulation from the NEW state
            //   four pipeline stages per lane-group, one stage per cycle.
            //   The state read is two cycles deep, so S_old(g) is on lane_q
            //   during p2 (address presented at p0); k*e and beta*(k*e) are
            //   independent of the state read and fill p1/p2.
            //
            //   p0 : address of group g; prod = q*S_new(g-1); o += prod(g-2);
            //        latch k[i]
            //   p1 : ke = k[i]*e[j]; latch q[i]
            //   p2 : aS = (alpha*S_old(g))>>16, bke = (beta*ke)>>31
            //   p3 : upd = aS+bke, saturate, write back, latch for the next group
            //-----------------------------------------------------------------
            ST_P4: begin
                case (sub[1:0])
                2'd0: begin
                    // p0
                    for (l = 0; l < 16; l = l + 1) begin
                        if (prod_vld) begin
                            o_mac[prod_grp][l] <= o_mac[prod_grp][l] + prod_r[l];
                        end
                        prod_r[l] <= prod_w[l];
                    end
                    prod_grp <= sub[3:2] - 2'd1;
                    prod_vld <= !((row_cnt == 6'd0) && (sub == 4'd0));
                    k_cur    <= k_buf[row_cnt];
                end
                2'd1: begin
                    // p1 : k[i] * e[j]
                    for (l = 0; l < 16; l = l + 1) begin
                        ke_r[l] <= ke_w[l];
                    end
                    q_cur <= q_buf[row_cnt];
                end
                2'd2: begin
                    // p2 : alpha*S_old(g) from lane_q, and the two halves of
                    //      beta*(k*e)
                    for (l = 0; l < 16; l = l + 1) begin
                        aS_r[l]     <= aS_w[l];
                        bke_hi_r[l] <= bke_hi_w[l];
                        bke_lo_r[l] <= bke_lo_w[l];
                    end
                end
                default: begin
                    // p3 : sum, saturate, write back, latch for the next group
                    for (l = 0; l < 16; l = l + 1) begin
                        snew_r[l] <= p4_snew[l];
                    end
                end
                endcase

                if (sub == 4'd15) begin
                    sub <= 4'd0;
                    if (row_cnt == 6'd63) begin
                        row_cnt <= 6'd0;
                        state   <= ST_P4D;
                    end else begin
                        row_cnt <= row_cnt + 6'd1;
                    end
                end else begin
                    sub <= sub + 4'd1;
                end
            end

            //-----------------------------------------------------------------
            // Phase 4 drain : the last two products and their accumulation
            //-----------------------------------------------------------------
            ST_P4D: begin
                if (sub == 4'd0) begin
                    for (l = 0; l < 16; l = l + 1) begin
                        if (prod_vld) begin
                            o_mac[prod_grp][l] <= o_mac[prod_grp][l] + prod_r[l];
                        end
                        prod_r[l] <= prod_w[l];
                    end
                    prod_grp <= 2'd3;
                    prod_vld <= 1'b1;
                    sub      <= 4'd1;
                end else begin
                    for (l = 0; l < 16; l = l + 1) begin
                        o_mac[2'd3][l] <= o_mac[2'd3][l] + prod_r[l];
                    end
                    prod_vld <= 1'b0;
                    sub      <= 4'd0;
                    out_cnt  <= 6'd0;
                    state    <= ST_OQ;
                end
            end

            //-----------------------------------------------------------------
            // Output quantisation : Q6.26 -> Q2.14, registered once per token
            //-----------------------------------------------------------------
            ST_OQ: begin
                for (g = 0; g < 4; g = g + 1) begin
                    for (l = 0; l < 16; l = l + 1) begin
                        o_q_r[g*16+l] <= sat16(o_mac[g][l], 12);
                    end
                end
                out_cnt  <= 6'd0;
                tvalid_r <= 1'b0;
                state    <= ST_OUT;
            end

            //-----------------------------------------------------------------
            // Phase 5 : emit 64 beats of o[]
            //-----------------------------------------------------------------
            ST_OUT: begin
                if (!tvalid_r || m_axis_tready) begin
                    tdata_r  <= {15'd0, (out_cnt == 6'd63), o_q_r[out_cnt]};
                    tlast_r  <= (out_cnt == 6'd63);
                    tvalid_r <= 1'b1;
                    if (out_cnt == 6'd63) begin
                        out_cnt <= 6'd0;
                        state   <= ST_IDLE;
                    end else begin
                        out_cnt <= out_cnt + 6'd1;
                    end
                end
            end

            default: begin
                state    <= ST_IDLE;
                tready_r <= 1'b0;
                tvalid_r <= 1'b0;
            end
            endcase
        end
    end

endmodule
