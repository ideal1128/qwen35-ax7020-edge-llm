//=============================================================================
// gdn_core.v  --  Gated DeltaNet in-place state recurrence accelerator
//
// Target : Xilinx Zynq-7020 (XC7Z020-2CLG400I), 180 MHz
// Style  : Verilog-2001, single clock domain (aclk), async active-low reset
//
//---------------------------------------------------------------------------
// STATE STORAGE / BANK LAYOUT  (judgement call, documented here)
//---------------------------------------------------------------------------
// 8 heads x 64 rows x 64 cols x 16 bit = 64 KB, split into 16 parallel banks
// so that the 16 MAC lanes each hit a different bank every cycle.
//
//   element (h, i, j)  ->  bank  = j % 16        (== lane index)
//                          word  = (h*64 + i)*4 + (j/16)
//
// Each bank is its own flat 1-D array of 2048 words of 16 bit (bank0..bank15,
// declared individually rather than as one 2-D unpacked array so that BRAM
// inference is unambiguous).  A 64-element row is therefore 4 words per lane,
// and one row is processed in 4 lane-groups of 16 parallel MACs.
//
// Both passes over the state use the same port:
//   pass 1 (phase 2) : u[j] += k[i] * S[i][j]        (read only)
//   pass 2 (phase 4) : S[i][j] = alpha*S + beta*k*e   (read + write back)
//                      o[j]   += q[i] * S_new
//
// Each lane-group costs 2 cycles: one to present the address (synchronous BRAM
// read) and one to consume the registered data.  One row = 8 cycles, one pass
// = 512 cycles, one token = 1024 cycles plus phase 3 and the 64 output beats.
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
//---------------------------------------------------------------------------
// MODE HANDLING (judgement call)
//---------------------------------------------------------------------------
// mode==0 (decode) is level sensitive and free running: after the 64 output
// beats the core returns to IDLE and immediately re-arms for the next token,
// which is what a streaming inference loop wants.
// mode==1 (reset) and mode==2 (dump) are re-triggered only when the mode input
// changes value, so a mode held high does not loop forever.  The mode input is
// sampled while idle; changing it during an operation has no effect until the
// operation completes.
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
    // Control
    //-------------------------------------------------------------------------
    reg [3:0]  state;
    reg [2:0]  sub;          // 0..7 inside a row: {lane group, read/consume}
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
    wire [10:0] grp_addr  = base_addr + {9'd0, sub[2:1]};          // + j/16
    wire [10:0] dump_addr = {head_sel[2:0], dump_cnt[11:6], 2'b00}
                            + {9'd0, dump_cnt[5:4]};
    wire [10:0] rst_base  = (head_sel < 8'd8) ? {head_sel[2:0], 8'd0} : 11'd0;
    wire [10:0] rst_last  = (head_sel < 8'd8) ? 11'd255 : 11'd2047;
    wire [10:0] rst_addr  = rst_base + rst_cnt;

    wire [10:0] bank_addr = (state == ST_DUMP) ? dump_addr :
                            (state == ST_RST)  ? rst_addr  : grp_addr;

    wire bank_we = (state == ST_RST) || ((state == ST_P4) && sub[0]);

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
    // Per-lane datapath
    //   mul_a/mul_s : one shared 16x16 multiplier per lane
    //                 phase 2 -> k[i] * S          (Q6.26)
    //                 phase 4 -> alpha * S          (Q4.28)
    //   p2_prod     : phase 2 product
    //   p4_snew     : phase 4 updated state, Q4.12
    //   p4_prod     : q[i] * S_new    (Q6.26)
    //-------------------------------------------------------------------------
    wire signed [15:0] mul_a = (state == ST_P2) ? k_cur
                                                : $signed({1'b0, alpha_u});
    wire signed [31:0] p2_prod [0:15];
    wire signed [15:0] p4_snew [0:15];
    wire signed [31:0] p4_prod [0:15];

    genvar gi;
    generate
        for (gi = 0; gi < 16; gi = gi + 1) begin : g_lane
            localparam [3:0] LID = gi;
            wire [5:0] j_idx = {sub[2:1], LID};

            // shared multiplier: k*S in phase 2, alpha*S in phase 4
            wire signed [31:0] mul_s = $signed(mul_a) * $signed(lane_rd[gi]);

            // phase 2 term
            assign p2_prod[gi] = mul_s;

            // phase 4 : S_new = sat(alpha*S + beta*k*e)
            wire signed [31:0] ke  = $signed(k_cur) * $signed(e_buf[j_idx]); // Q5.27
            wire signed [47:0] bke = $signed({1'b0, beta_u}) * ke;           // Q5.43
            wire signed [31:0] upd = (mul_s >>> 16) + (bke >>> 31);          // Q4.12

            assign p4_snew[gi] = sat16(upd, 0);
            assign p4_prod[gi] = $signed(q_cur) * $signed(p4_snew[gi]);      // Q6.26
        end
    endgenerate

    //-------------------------------------------------------------------------
    // Output quantisation, Q6.26 -> Q2.14
    //-------------------------------------------------------------------------
    wire signed [15:0] o_q [0:63];

    genvar gk;
    generate
        for (gk = 0; gk < 64; gk = gk + 1) begin : g_oq
            localparam [1:0] GG = gk / 16;
            localparam [3:0] LL = gk % 16;
            assign o_q[gk] = sat16(o_mac[GG][LL], 12);
        end
    endgenerate

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
            sub        <= 3'd0;
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
            for (g = 0; g < 64; g = g + 1) begin
                q_buf[g] <= 16'sd0;
                k_buf[g] <= 16'sd0;
                v_buf[g] <= 16'sd0;
                u_q[g]   <= 16'sd0;
                e_buf[g] <= 16'sd0;
            end
        end else begin
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
                        sub      <= 3'd0;
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
            // Phase 2 : u = k^T * S, one row per 8 cycles
            //-----------------------------------------------------------------
            ST_P2: begin
                if (sub == 3'd0) begin
                    k_cur <= k_buf[row_cnt];
                end
                if (sub[0]) begin
                    for (l = 0; l < 16; l = l + 1) begin
                        u_mac[sub[2:1]][l] <= u_mac[sub[2:1]][l] + p2_prod[l];
                    end
                end
                if (sub == 3'd7) begin
                    sub <= 3'd0;
                    if (row_cnt == 6'd63) begin
                        row_cnt <= 6'd0;
                        state   <= ST_P3;
                    end else begin
                        row_cnt <= row_cnt + 6'd1;
                    end
                end else begin
                    sub <= sub + 3'd1;
                end
            end

            //-----------------------------------------------------------------
            // Phase 3 : u_quant then e[j] = v[j] - u_quant
            //-----------------------------------------------------------------
            ST_P3: begin
                if (sub == 3'd0) begin
                    for (g = 0; g < 4; g = g + 1) begin
                        for (l = 0; l < 16; l = l + 1) begin
                            u_q[g*16+l] <= sat16(u_mac[g][l], 12);
                        end
                    end
                    sub <= 3'd1;
                end else begin
                    for (g = 0; g < 4; g = g + 1) begin
                        for (l = 0; l < 16; l = l + 1) begin
                            e_buf[g*16+l] <=
                                sat16({{16{v_buf[g*16+l][15]}}, v_buf[g*16+l]} -
                                      {{16{u_q[g*16+l][15]}},   u_q[g*16+l]}, 0);
                            o_mac[g][l] <= 32'sd0;
                        end
                    end
                    sub     <= 3'd0;
                    row_cnt <= 6'd0;
                    state   <= ST_P4;
                end
            end

            //-----------------------------------------------------------------
            // Phase 4 : in-place update + o accumulation from the NEW state
            //-----------------------------------------------------------------
            ST_P4: begin
                if (sub == 3'd0) begin
                    q_cur <= q_buf[row_cnt];
                end
                if (sub[0]) begin
                    for (l = 0; l < 16; l = l + 1) begin
                        o_mac[sub[2:1]][l] <= o_mac[sub[2:1]][l] + p4_prod[l];
                    end
                end
                if (sub == 3'd7) begin
                    sub <= 3'd0;
                    if (row_cnt == 6'd63) begin
                        row_cnt <= 6'd0;
                        out_cnt <= 6'd0;
                        state   <= ST_OUT;
                    end else begin
                        row_cnt <= row_cnt + 6'd1;
                    end
                end else begin
                    sub <= sub + 3'd1;
                end
            end

            //-----------------------------------------------------------------
            // Phase 5 : emit 64 beats of o[]
            //-----------------------------------------------------------------
            ST_OUT: begin
                if (!tvalid_r || m_axis_tready) begin
                    tdata_r  <= {15'd0, (out_cnt == 6'd63), o_q[out_cnt]};
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
                state   <= ST_IDLE;
                tready_r <= 1'b0;
                tvalid_r <= 1'b0;
            end
            endcase
        end
    end

endmodule
