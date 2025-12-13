// Copyright 2021 The CFU-Playground Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
`include "TPU.v"
`include "global_buffer_bram.v"


module Cfu (
  input               cmd_valid,
  output              cmd_ready,
  input      [9:0]    cmd_payload_function_id,
  input      [31:0]   cmd_payload_inputs_0,
  input      [31:0]   cmd_payload_inputs_1,
  output     reg         rsp_valid,
  input               rsp_ready,
  output    reg [31:0]   rsp_payload_outputs_0,
  input               reset,
  input               clk
);

  // Trivial handshaking for a combinational CFU
  // assign rsp_valid = cmd_valid;
  // assign cmd_ready = rsp_ready;


  wire busy;
  wire rst_n = ~reset;

  wire [8:0] K = cmd_payload_inputs_1[15:0];
  wire [8:0] M = cmd_payload_inputs_0[15:0];
  wire [8:0] N = cmd_payload_inputs_0[31:16];
  reg [1:0] mode;

  wire            in_valid_TPU = cmd_valid && cmd_ready && (cmd_payload_function_id == 8);
  
  wire            A_wr_en_TPU;
  wire [15:0]     A_index_TPU;
  wire [31:0]     A_data_in_TPU;

  wire            B_wr_en_TPU;
  wire [15:0]     B_index_TPU;
  wire [31:0]     B_data_in_TPU;

  wire            C_wr_en_TPU;
  wire [15:0]     C_index_TPU;
  wire [127:0]    C_data_in_TPU;


  reg            A_wr_en_CFU;
  reg [15:0]     A_index_CFU;
  reg [31:0]     A_data_in_CFU;

  reg            B_wr_en_CFU;
  reg [15:0]     B_index_CFU;
  reg [31:0]     B_data_in_CFU;

  reg            C_wr_en_CFU;
  reg [15:0]     C_index_CFU;
  reg [127:0]    C_data_in_CFU;


  wire            A_wr_en = (mode==2) ? A_wr_en_TPU : A_wr_en_CFU;
  wire [15:0]     A_index = (mode==2) ? A_index_TPU : A_index_CFU;
  wire [31:0]     A_data_in = (mode==2) ? A_data_in_TPU : A_data_in_CFU;
  wire [31:0]     A_data_out;
  wire            B_wr_en = (mode==2) ? B_wr_en_TPU : B_wr_en_CFU;
  wire [15:0]     B_index = (mode==2) ? B_index_TPU : B_index_CFU;
  wire [31:0]     B_data_in = (mode==2) ? B_data_in_TPU : B_data_in_CFU;
  wire [31:0]     B_data_out;
  wire            C_wr_en = (mode==2) ? C_wr_en_TPU : C_wr_en_CFU;
  wire [15:0]     C_index = (mode==2) ? C_index_TPU : C_index_CFU;
  wire [127:0]     C_data_in = (mode==2) ? C_data_in_TPU : C_data_in_CFU;
  wire [127:0]     C_data_out;
  
  wire ram_en = (mode!=0);

  //
  // select output -- note that we're not fully decoding the 3 function_id bits
  //
  /*assign rsp_payload_outputs_0 = cmd_payload_function_id[0] ? 
                                           cmd_payload_inputs_1 :
                                           cmd_payload_inputs_0 ;*/

  reg [1:0] offset;
  assign cmd_ready = ~rsp_valid;                                         
  always @(posedge clk) begin
    if (reset) begin
      rsp_valid <= 1'b0;
      mode <= 2'b0;
      offset <= 2'b0;
      rsp_payload_outputs_0 <= 32'b0;
      A_data_in_CFU <= 0;
      A_index_CFU <= 0;
      A_wr_en_CFU <= 0;
      B_data_in_CFU <= 0;
      B_index_CFU <= 0;
      B_wr_en_CFU <= 0;
      C_data_in_CFU <= 0;
      C_index_CFU <= 0;
      C_wr_en_CFU <= 0;
    end else if (mode != 2'b0) begin
      if((mode == 2'b1)||(mode == 2'd3)) begin
        rsp_valid <= 1'b1;
        mode <= 2'b0;
        if(mode == 2'd3) begin
          rsp_payload_outputs_0 <= (offset == 2'd3) ? C_data_out[31:0]
                                  :(offset == 2'd2) ? C_data_out[63:32]
                                  :(offset == 2'd1) ? C_data_out[95:64]
                                  :(offset == 2'd0) ? C_data_out[127:96]
                                  : 0;
        end else begin
          A_wr_en_CFU <= 0;
          B_wr_en_CFU <= 0;
          rsp_payload_outputs_0 <= 0;
        end
      end else begin
        rsp_valid <= ~busy;
        mode <= (busy) ? mode : 2'b0;
        rsp_payload_outputs_0 <= C_index;
      end
    end else if (rsp_valid) begin
      // Waiting to hand off response to CPU.
      rsp_valid <= ~rsp_ready;
    end else if (cmd_valid) begin
      // Accumulate step:
      if(cmd_payload_function_id == 0) begin
        mode <= 2'b1;
        if(cmd_payload_inputs_1[24]==0) begin
          A_wr_en_CFU <= 1;
          A_data_in_CFU <= cmd_payload_inputs_0;
          A_index_CFU <= cmd_payload_inputs_1[15:0];
        end else begin
          B_wr_en_CFU <= 1;
          B_data_in_CFU <= cmd_payload_inputs_0;
          B_index_CFU <= cmd_payload_inputs_1[15:0];
        end
      end else if(cmd_payload_function_id == 8) begin
        mode <= 2'd2;
      end else if(cmd_payload_function_id == 16) begin
        mode <= 2'd3;
        offset <= cmd_payload_inputs_1[1:0];
        C_index_CFU <= cmd_payload_inputs_0[15:0];
      end else begin
        rsp_valid <= 1'b1;
        rsp_payload_outputs_0 <= {22'b0,cmd_payload_function_id};
      end
    end
  end

  TPU My_TPU(
    .clk            (clk),     
    .rst_n          (rst_n),     
    .in_valid       (in_valid_TPU),         
    .K              (K), 
    .M              (M), 
    .N              (N), 
    .busy           (busy),     
    .A_wr_en        (A_wr_en_TPU),         
    .A_index        (A_index_TPU),         
    .A_data_in      (A_data_in_TPU),         
    .A_data_out     (A_data_out),         
    .B_wr_en        (B_wr_en_TPU),         
    .B_index        (B_index_TPU),         
    .B_data_in      (B_data_in_TPU),         
    .B_data_out     (B_data_out),         
    .C_wr_en        (C_wr_en_TPU),         
    .C_index        (C_index_TPU),         
    .C_data_in      (C_data_in_TPU),         
    .C_data_out     (C_data_out)         
  );

   global_buffer_bram #(
      .ADDR_BITS(14),
      .DATA_BITS(32)
  )
  gbuff_A(
      .clk(clk),
      .rst_n(rst_n),
      .ram_en(ram_en),
      .wr_en(A_wr_en),
      .index(A_index),
      .data_in(A_data_in),
      .data_out(A_data_out)
  );

   global_buffer_bram #(
      .ADDR_BITS(14),
      .DATA_BITS(32)
  ) gbuff_B(
      .clk(clk),
      .rst_n(rst_n),
      .ram_en(ram_en),
      .wr_en(B_wr_en),
      .index(B_index),
      .data_in(B_data_in),
      .data_out(B_data_out)
  );


   global_buffer_bram #(
      .ADDR_BITS(14),
      .DATA_BITS(128)
  ) gbuff_C(
      .clk(clk),
      .rst_n(rst_n),
      .ram_en(ram_en),
      .wr_en(C_wr_en),
      .index(C_index),
      .data_in(C_data_in),
      .data_out(C_data_out)
  );
endmodule
