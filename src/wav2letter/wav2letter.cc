#include "wav2letter.h" // Your .h file
#include <stdio.h>
#include "menu.h"
#include "tflite.h"

#include <cstring>
// 1. INCLUDE YOUR MODEL AND TEST DATA
#include "model/wav2letter_pruned_int8.h"
#include "test_data/test_input.h"
#include "test_data/test_output.h"  

#include "playground_util/console.h"

// Initialize everything once
static void wav2letter_pruned_init(void) {
  tflite_load_model(wav2letter_pruned_int8, wav2letter_pruned_int8_len);
}



static void do_golden_tests() {
  printf("Running golden test...\n");

  // 1. Set input
  printf("Setting model input...\n");
  tflite_set_input(g_test_input_data);

  // 2. Run inference
  printf("Running inference...\n");
  tflite_classify();

  // 3. Get the output tensor
  int8_t* output = tflite_get_output();
  printf("Inference complete, comparing output...\n");

  // 4. Compare the model's output with your golden data
  // We use memcmp to compare the raw bytes of the two arrays.
  if (memcmp(output, g_test_output_data, g_test_output_data_len) != 0) {
    printf("*** FAIL: Golden test failed.\n");
    
    // Print first 4 bytes for debugging, just like the imgc example
    printf("First 4 bytes actual:   %d %d %d %d\n", 
           output[0], output[1], output[2], output[3]);
    printf("First 4 bytes expected: %d %d %d %d\n", 
           g_test_output_data[0], g_test_output_data[1], 
           g_test_output_data[2], g_test_output_data[3]);
  } else {
    printf("OK   Golden tests passed!\n");
  }
}

// 4. ADD YOUR NEW TEST TO THE MENU
static struct Menu MENU = {
    "Tests for wav2letter_pruned",
    "wav2letter",
    {
        MENU_ITEM('g', "Run golden tests", do_golden_tests),
        MENU_END,
    },
};

// For integration into menu system
void wav2letter_pruned_menu() {
  wav2letter_pruned_init();
  menu_run(&MENU);
}