// ============================================================================
// fwht64.v -- 64-point Fast Walsh-Hadamard Transform (natural / Hadamard order)
// ----------------------------------------------------------------------------
//   z[i] = (1/8) * sum_{j=0..63} (-1)^popcount(i & j) * a[j]
//
// Structure:
//   * 6 radix-2 butterfly stages in natural order. At stage s, every index i
//     with bit s clear is paired with j = i | (1<<s) and replaced by
//     (a[i]+a[j], a[i]-a[j]).
//   * Serialised engine, one butterfly per cycle: 6 stages * 32 butterflies
//     = 192 cycles, over a 64-word array of 22-bit signed registers.
//   * Scaling / saturation: the worst-case magnitude after all 6 stages is
//     64 * 32768 = 2097152, which fits a 22-bit signed word exactly, so the
//     accumulators are carried at full width through every stage with no
//     intermediate scaling. After the last stage each result is arithmetic
//     shifted right by 3 (acc >>> 3, i.e. the 1/8 gain) and then saturated to
//     16-bit signed: > 32767 -> 32767, < -32768 -> -32768.
//
// Latency (aclk cycles):
//   first input element -> first output element : 257
//   last  input element -> last  output element : 257
//   block-to-block period (back-to-back blocks) : 320 (64 load + 192 + 64 out)
//   bypass path latency                         : 1
//
// Handshake:
//   * Elements are accepted only during the load window (state LOAD). in_valid
//     is ignored while the butterfly engine or the output stage runs, so a
//     producer must not present the next block until this block is emitted.
//   * in_last is expected on the 64th element. It is also used for resync: an
//     early in_last discards the partial block and restarts loading at index 0.
//   * bypass = 1 : output follows input through a single register (1 cycle),
//     block / last alignment is preserved, and any block in progress is
//     aborted so the next non-bypass block starts clean.
//
// Verilog-2001, single clock aclk, active-low async reset on all control flops.
// ============================================================================

module fwht64 (
    input  wire                aclk,
    input  wire                aresetn,
    input  wire                in_valid,
    input  wire signed [15:0]  in_data,
    input  wire                in_last,
    output wire                out_valid,
    output wire signed [15:0]  out_data,
    output wire                out_last,
    input  wire                bypass
);

    localparam [1:0] ST_LOAD = 2'd0;
    localparam [1:0] ST_BFLY = 2'd1;
    localparam [1:0] ST_OUT  = 2'd2;

    reg signed [21:0] acc_r [0:63];

    reg [1:0]  state;
    reg [5:0]  wr_ptr;
    reg [2:0]  stage;
    reg [4:0]  bfly_idx;
    reg [5:0]  rd_ptr;

    reg                out_valid_r;
    reg signed [15:0]  out_data_r;
    reg                out_last_r;

    // ------------------------------------------------------------------
    // Butterfly index generation: insert a 0 bit at position `stage`
    // into the 5-bit butterfly counter, giving i (bit s clear) and
    // j = i | (1<<s).
    // ------------------------------------------------------------------
    wire [5:0] bf_mask = (6'd1 << stage) - 6'd1;
    wire [5:0] bf_lo   = {1'b0, bfly_idx} & bf_mask;
    wire [5:0] bf_hi   = ({1'b0, bfly_idx} & ~bf_mask) << 1;
    wire [5:0] bf_i    = bf_hi | bf_lo;
    wire [5:0] bf_j    = bf_i | (6'd1 << stage);

    wire signed [21:0] bf_a   = acc_r[bf_i];
    wire signed [21:0] bf_b   = acc_r[bf_j];
    wire signed [21:0] bf_sum = bf_a + bf_b;
    wire signed [21:0] bf_dif = bf_a - bf_b;

    // ------------------------------------------------------------------
    // Final 1/8 scaling and 16-bit saturation
    // ------------------------------------------------------------------
    wire signed [21:0] acc    = acc_r[rd_ptr];
    wire signed [21:0] scaled = acc >>> 3;
    wire signed [15:0] sat    = (scaled > 22'sd32767)  ? 16'sh7fff :
                                (scaled < -22'sd32768) ? 16'sh8000 :
                                scaled[15:0];

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            state       <= ST_LOAD;
            wr_ptr      <= 6'd0;
            stage       <= 3'd0;
            bfly_idx    <= 5'd0;
            rd_ptr      <= 6'd0;
            out_valid_r <= 1'b0;
            out_data_r  <= 16'sd0;
            out_last_r  <= 1'b0;
        end else begin
            out_valid_r <= 1'b0;
            out_last_r  <= 1'b0;
            out_data_r  <= 16'sd0;

            if (bypass) begin
                // Transparent path: one register only, alignment kept.
                out_valid_r <= in_valid;
                out_data_r  <= in_data;
                out_last_r  <= in_last;
                state       <= ST_LOAD;
                wr_ptr      <= 6'd0;
                stage       <= 3'd0;
                bfly_idx    <= 5'd0;
                rd_ptr      <= 6'd0;
            end else begin
                case (state)

                    // ---- collect one 64-element block -------------------
                    ST_LOAD: begin
                        if (in_valid) begin
                            acc_r[wr_ptr] <= {{6{in_data[15]}}, in_data};
                            if (wr_ptr == 6'd63) begin
                                wr_ptr   <= 6'd0;
                                stage    <= 3'd0;
                                bfly_idx <= 5'd0;
                                state    <= ST_BFLY;
                            end else if (in_last) begin
                                // early in_last: drop partial block, realign
                                wr_ptr <= 6'd0;
                            end else begin
                                wr_ptr <= wr_ptr + 6'd1;
                            end
                        end
                    end

                    // ---- 192 butterflies, one per cycle -----------------
                    ST_BFLY: begin
                        acc_r[bf_i] <= bf_sum;
                        acc_r[bf_j] <= bf_dif;
                        if (bfly_idx == 5'd31) begin
                            bfly_idx <= 5'd0;
                            if (stage == 3'd5) begin
                                rd_ptr <= 6'd0;
                                state  <= ST_OUT;
                            end else begin
                                stage <= stage + 3'd1;
                            end
                        end else begin
                            bfly_idx <= bfly_idx + 5'd1;
                        end
                    end

                    // ---- scale, saturate and emit 64 results ------------
                    ST_OUT: begin
                        out_valid_r <= 1'b1;
                        out_data_r  <= sat;
                        out_last_r  <= (rd_ptr == 6'd63);
                        if (rd_ptr == 6'd63) begin
                            rd_ptr <= 6'd0;
                            state  <= ST_LOAD;
                        end else begin
                            rd_ptr <= rd_ptr + 6'd1;
                        end
                    end

                    default: begin
                        state <= ST_LOAD;
                    end
                endcase
            end
        end
    end

    assign out_valid = out_valid_r;
    assign out_data  = out_data_r;
    assign out_last  = out_last_r;

endmodule
