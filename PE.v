module PE(
    input           in_valid,
    input           clk,
    input           rst_n,
    input [7:0]     req_A0,
    input [7:0]     req_A1,
    input [7:0]     req_A2,
    input [7:0]     req_A3,
    input [7:0]     req_B0,
    input [7:0]     req_B1,
    input [7:0]     req_B2,
    input [7:0]     req_B3,
    output [127:0]  req_C0,
    output [127:0]  req_C1,
    output [127:0]  req_C2,
    output [127:0]  req_C3,
    input           clear
);

reg [31:0] inputA0;
reg [31:0] inputA1;
reg [31:0] inputA2;
reg [31:0] inputA3;
reg [31:0] inputB0;
reg [31:0] inputB1;
reg [31:0] inputB2;
reg [31:0] inputB3;


reg signed [127:0] outputC0;
reg signed [127:0] outputC1;
reg signed [127:0] outputC2;
reg signed [127:0] outputC3;

wire signed [31:0] PE00 = $signed(inputA0[7:0])    * $signed(inputB0[7:0]);
wire signed [31:0] PE01 = $signed(inputA0[15:8])   * $signed(inputB1[7:0]);
wire signed [31:0] PE02 = $signed(inputA0[23:16])  * $signed(inputB2[7:0]);
wire signed [31:0] PE03 = $signed(inputA0[31:24])  * $signed(inputB3[7:0]);
wire signed [31:0] PE10 = $signed(inputA1[7:0])    * $signed(inputB0[15:8]);
wire signed [31:0] PE11 = $signed(inputA1[15:8])   * $signed(inputB1[15:8]);
wire signed [31:0] PE12 = $signed(inputA1[23:16])  * $signed(inputB2[15:8]);
wire signed [31:0] PE13 = $signed(inputA1[31:24])  * $signed(inputB3[15:8]);
wire signed [31:0] PE20 = $signed(inputA2[7:0])    * $signed(inputB0[23:16]);
wire signed [31:0] PE21 = $signed(inputA2[15:8])   * $signed(inputB1[23:16]);
wire signed [31:0] PE22 = $signed(inputA2[23:16])  * $signed(inputB2[23:16]);
wire signed [31:0] PE23 = $signed(inputA2[31:24])  * $signed(inputB3[23:16]);
wire signed [31:0] PE30 = $signed(inputA3[7:0])    * $signed(inputB0[31:24]);
wire signed [31:0] PE31 = $signed(inputA3[15:8])   * $signed(inputB1[31:24]);
wire signed [31:0] PE32 = $signed(inputA3[23:16])  * $signed(inputB2[31:24]);
wire signed [31:0] PE33 = $signed(inputA3[31:24])  * $signed(inputB3[31:24]);


assign req_C0 = outputC0;
assign req_C1 = outputC1;
assign req_C2 = outputC2;
assign req_C3 = outputC3;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        inputA0 <= 0;
        inputA1 <= 0;
        inputA2 <= 0;
        inputA3 <= 0;
        inputB0 <= 0;
        inputB1 <= 0;
        inputB2 <= 0;
        inputB3 <= 0;
        outputC0 <= 0;
        outputC1 <= 0;
        outputC2 <= 0;
        outputC3 <= 0;
    end
    else if( clear ) begin
        inputA0 <= 0;
        inputA1 <= 0;
        inputA2 <= 0;
        inputA3 <= 0;
        inputB0 <= 0;
        inputB1 <= 0;
        inputB2 <= 0;
        inputB3 <= 0;
        outputC0 <= 0;
        outputC1 <= 0;
        outputC2 <= 0;
        outputC3 <= 0;
    end else begin
        inputA0 <= {inputA0[23:0], req_A0};
        inputA1 <= {inputA1[23:0], req_A1};
        inputA2 <= {inputA2[23:0], req_A2};
        inputA3 <= {inputA3[23:0], req_A3};
        inputB0 <= {inputB0[23:0], req_B0};
        inputB1 <= {inputB1[23:0], req_B1};
        inputB2 <= {inputB2[23:0], req_B2};
        inputB3 <= {inputB3[23:0], req_B3};
        outputC0 <= {$signed(req_C0[127:96]) + $signed(PE00), $signed(req_C0[95:64]) + $signed(PE01), $signed(req_C0[63:32]) + $signed(PE02), $signed(req_C0[31:0]) + $signed(PE03)};
        outputC1 <= {$signed(req_C1[127:96]) + $signed(PE10), $signed(req_C1[95:64]) + $signed(PE11), $signed(req_C1[63:32]) + $signed(PE12), $signed(req_C1[31:0]) + $signed(PE13)};
        outputC2 <= {$signed(req_C2[127:96]) + $signed(PE20), $signed(req_C2[95:64]) + $signed(PE21), $signed(req_C2[63:32]) + $signed(PE22), $signed(req_C2[31:0]) + $signed(PE23)};
        outputC3 <= {$signed(req_C3[127:96]) + $signed(PE30), $signed(req_C3[95:64]) + $signed(PE31), $signed(req_C3[63:32]) + $signed(PE32), $signed(req_C3[31:0]) + $signed(PE33)};
    end

end
endmodule