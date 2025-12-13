`include "PE.v"
module SYSTOLIC_ARRAY(
    input                 clk,
    input                 rst_n,
    input                 in_valid,
    input       [8:0]     K,
    input       [8:0]     M,
    input       [8:0]     N,
    output  reg           busy,

    output                A_wr_en,
    output      [15:0]    A_index,
    output      [31:0]    A_data_in,
    input       [31:0]    A_data_out,

    output                B_wr_en,
    output      [15:0]    B_index,
    output      [31:0]    B_data_in,
    input       [31:0]    B_data_out,

    output  reg           C_wr_en,
    output  reg [15:0]    C_index,
    output      [127:0]   C_data_in,
    input       [127:0]   C_data_out
);
    
    reg [8:0] M_reg;
    reg [8:0] N_reg;
    reg [8:0] K_reg;
    reg [2:0] C_data_in_sel;
    reg load_en;
    reg [2:0] cnt;
    reg [15:0] index;
    reg [15:0] baseA;
    reg [15:0] baseB;

    reg [31:0] A_0;
    reg [31:0] A_1;
    reg [31:0] A_2;
    reg [31:0] A_3;
    reg [31:0] B_0;
    reg [31:0] B_1;
    reg [31:0] B_2;
    reg [31:0] B_3;

    reg [15:0] baseA_K;
    reg [15:0] baseB_K;

    assign A_wr_en = 1'b0;
    assign B_wr_en = 1'b0;
    assign A_data_in = 32'bx;
    assign B_data_in = 32'bx;
    assign A_index = index + baseA_K;
    assign B_index = index + baseB_K;

    wire [7:0]     req_A0 = A_0[7:0];
    wire [7:0]     req_A1 = A_1[7:0];
    wire [7:0]     req_A2 = A_2[7:0];
    wire [7:0]     req_A3 = A_3[7:0];
    wire [7:0]     req_B0 = B_0[7:0];
    wire [7:0]     req_B1 = B_1[7:0];
    wire [7:0]     req_B2 = B_2[7:0];
    wire [7:0]     req_B3 = B_3[7:0];
    wire [127:0]   req_C0;
    wire [127:0]   req_C1;
    wire [127:0]   req_C2;
    wire [127:0]   req_C3;

    

    reg [15:0] n; 
    reg [8:0] Nd4;
    reg [8:0] Nr4;
    reg [8:0] Md4;
    reg [8:0] Mr4;
    reg [15:0] A_len;
    reg  [2:0] limit;
    
    reg clear;

    assign C_data_in = (C_data_in_sel == 2'd0) ? req_C0
                     : (C_data_in_sel == 2'd1) ? req_C1
                     : (C_data_in_sel == 2'd2) ? req_C2
                     : (C_data_in_sel == 2'd3) ? req_C3
                     : 128'bx;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin 
            busy <= 1'b0;
            C_wr_en <= 1'b0;
            C_index <= 0;
            clear <= 1;
        end
        else if (in_valid) begin
            busy <= in_valid;
            
            M_reg <= M;
            N_reg <= N;
            K_reg <= K;

            A_0 <= 8'b0;
            A_1 <= 8'b0;
            A_2 <= 8'b0;
            A_3 <= 8'b0;
            B_0 <= 8'b0;
            B_1 <= 8'b0;
            B_2 <= 8'b0;
            B_3 <= 8'b0;
            C_wr_en <= 1'b0;
            load_en <= 1;
            C_data_in_sel <= 2'd0;
            C_index <= 0;
            index <= 0;
            baseA <= 0;
            baseB <= 0;
            cnt <= 5;
            load_en <= 1;
            clear <= 1;
            limit <= 3;
            Nd4 <= (N/4);
            Nr4 <= (N%4);
            Md4 <= M/4;
            Mr4 <= M%4;
            baseA_K <= 0;
            baseB_K <= 0;
        end else if (load_en) begin
            clear <= 0;
            n <= M_reg * (Nd4+(Nr4!=0));
            A_len <= Md4 + ((Mr4)!=0);
            if(index == (K_reg-1)) begin
                load_en <= 0;
                cnt <= 5;
                C_data_in_sel <= 2'd0;
                index <= 0;
                if( (baseA + 1) == A_len ) begin
                    baseA <= 0;
                    baseB <= baseB + 1;
                    baseB_K <= baseB_K + K_reg;
                    baseA_K <= 0;
                    limit <=(Mr4==0)?4:Mr4;
                end else begin
                    baseA <= baseA + 1;
                    limit <= 4;
                    baseA_K <= baseA_K + K_reg;
                end
            end else begin
                index <= index + 1;
            end
            A_0 <= A_0>>8 | A_data_out[31:24];
            A_1 <= A_1>>8 | A_data_out[23:16]<<8;
            A_2 <= A_2>>8 | A_data_out[15:8]<<16;
            A_3 <= A_3>>8 | A_data_out[7:0]<<24;
            B_0 <= B_0>>8 | B_data_out[31:24];
            B_1 <= B_1>>8 | B_data_out[23:16]<<8;
            B_2 <= B_2>>8 | B_data_out[15:8]<<16;
            B_3 <= B_3>>8 | B_data_out[7:0]<<24;
        end else if(cnt>0) begin
            cnt <= cnt - 1;
            A_0 <= A_0>>8;
            A_1 <= A_1>>8;
            A_2 <= A_2>>8;
            A_3 <= A_3>>8;
            B_0 <= B_0>>8;
            B_1 <= B_1>>8;
            B_2 <= B_2>>8;
            B_3 <= B_3>>8;
            C_wr_en <= (cnt==1);
        end else if(busy) begin
            C_wr_en <= ((C_data_in_sel+1) < limit); 
            C_index <= C_index + (C_wr_en);
            C_data_in_sel <= (C_data_in_sel < 4) ? C_data_in_sel + 1 : C_data_in_sel;
            busy <= (~(C_data_in_sel == 4)) || (C_index <(n)) ;
            load_en <= (C_data_in_sel == 4) && (C_index <(n)) ;
            clear <= (C_data_in_sel == 4);
        end
    end



    PE my_PE(
        .clk            (clk),     
        .rst_n          (rst_n),     
        .in_valid       (in_valid),
        .req_A0         (req_A0),
        .req_A1         (req_A1),
        .req_A2         (req_A2),
        .req_A3         (req_A3),
        .req_B0         (req_B0),
        .req_B1         (req_B1),
        .req_B2         (req_B2),
        .req_B3         (req_B3),
        .req_C0         (req_C0),
        .req_C1         (req_C1),
        .req_C2         (req_C2),
        .req_C3         (req_C3),
        .clear          (clear)
    );
endmodule