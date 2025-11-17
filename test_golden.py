import argparse
import serial
import time
import re
import os

# --- Configuration ---
INPUT_BIN_PATH = 'src/wav2letter/test_data/test_input.bin'
GOLDEN_BIN_PATH = 'src/wav2letter/test_data/test_output.bin'

# Model input shape (1, 296, 39)
INPUT_SIZE = 1 * 296 * 39  # 11544 bytes
# Model output shape (1, 1, 148, 29)
OUTPUT_SIZE = 1 * 1 * 148 * 29 # 4292 bytes (This is correct)

# Serial protocol
READY_MSG = 'm-ready\r\n'
SEND_BYTES_PER_CMD = 32 
# --- End Configuration ---


def parse_arg():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Wav2letter Functional Test Script")
    parser.add_argument("--port", nargs='?', default='/dev/ttyUSB0', type=str,
                        help='Device port, e.g, --port /dev/ttyUSB0.')
    parser.add_argument("-p", nargs='?', dest='port', type=str,
                        help='Device port, e.g, -p /dev/ttyUSB0.')
    return parser.parse_args()

def byte_to_int8(b):
    """Converts a raw byte (0-255) to a signed int8_t (-128 to 127)."""
    if b > 127:
        return b - 256
    return b

def load_golden_data(filepath):
    """Loads the golden output tensor and converts it to a list of int8_t."""
    print(f"Loading golden output from {filepath}...")
    if not os.path.exists(filepath):
        print(f"Error: Golden file not found at {filepath}")
        exit()
        
    with open(filepath, 'rb') as f:
        golden_bytes = f.read()
        if len(golden_bytes) != OUTPUT_SIZE:
            print(f"Error: Golden file is wrong size! Expected {OUTPUT_SIZE}, got {len(golden_bytes)}")
            exit()
        # Convert raw bytes to list of signed int8_t
        golden_results = [byte_to_int8(b) for b in golden_bytes]
    print(f"Golden data loaded ({len(golden_results)} values).")
    return golden_results

# Helper function to read and print messages
def read_and_print(com, until_msg):
    msg = com.read_until(until_msg.encode()).decode('utf-8', 'ignore')
    print(f"BOARD: {msg.strip()}")
    return msg

def main():
    args = parse_arg()
    port = str(args.port)

    # 1. Load golden data first (fail fast)
    golden_results = load_golden_data(GOLDEN_BIN_PATH)

    # 2. Open Serial Port
    try:
        # <<< Timeout is 900 seconds (15 min) for the long inference >>>
        com = serial.Serial(port=port, baudrate=1843200, timeout=900)
    except Exception as e:
        print(f"Error: Opening serial port {port} failed: {e}")
        exit()

    if not com.is_open:
        print(f"Error: Serial port {port} is not open.")
        exit()

    print(f"Serial port {port} opened. Starting test...")
    
    try:
        
        print("Sending '3' for Project Menu...")
        com.write(b'3\n') 
        read_and_print(com, 'project> ') # Wait for project menu
        time.sleep(0.1) 

        print("Sending 'b' for Benchmark Mode...")
        com.write(b'b\n') 
        
        read_and_print(com, READY_MSG) 
        print("Benchmark mode entered. Device is ready.")
        
        # 4. 'name' command
        com.write("name%".encode())
        output = read_and_print(com, READY_MSG) 

        # 5. Load Input Data
        if not os.path.exists(INPUT_BIN_PATH):
             print(f"Error: Input file not found at {INPUT_BIN_PATH}")
             raise FileNotFoundError
             
        with open(INPUT_BIN_PATH, 'rb') as test_input:
            com.read_all() # Clear buffer
            
            db_load_cmd = f"db load {INPUT_SIZE}%"
            print(f"SENDING: {db_load_cmd.strip()}")
            com.write(db_load_cmd.encode())
            read_and_print(com, READY_MSG)

            print(f"Sending {INPUT_SIZE} bytes of input data...")
            bytes_sent = 0
            data_chunk = test_input.read(SEND_BYTES_PER_CMD)
            while len(data_chunk) > 0:
                db_data_cmd = f"db {data_chunk.hex()}%"
                com.write(db_data_cmd.encode())
                # Don't print the 361 "m-ready" messages
                com.read_until(READY_MSG.encode()) 
                bytes_sent += len(data_chunk)
                data_chunk = test_input.read(SEND_BYTES_PER_CMD)
            print(f"Finished sending {bytes_sent} bytes.")

        # 6. Run Inference
        infer_cmd = "infer 1 0%"
        print(f"SENDING: {infer_cmd.strip()}")
        com.write(infer_cmd.encode())
        
        # <<< MODIFIED: This is the new byte-by-byte loop to print dots >>>
        print("... Inference running (will print dots as they arrive):")
        msg_buffer_bytes = b''
        
        while True:
            # Read one byte at a time
            byte_char = com.read(1)
            
            # 1. Check for timeout (com.read() returns b'')
            if not byte_char:
                print("\nError: Read timed out after 15 minutes.")
                msg = msg_buffer_bytes.decode('utf-8', 'ignore') # return what we have
                break # Exit the loop
                
            # 2. Print the character live
            try:
                print(byte_char.decode('utf-8', 'ignore'), end='', flush=True)
            except:
                pass # Ignore decode errors
                
            # 3. Add to buffer and check for end message
            msg_buffer_bytes += byte_char
            if msg_buffer_bytes.endswith(READY_MSG.encode()):
                msg = msg_buffer_bytes.decode('utf-8', 'ignore')
                break # Success!
        
        print("\n... Inference complete.")
        # <<< END MODIFICATION >>>

        # 7. Parse Latency
        try:
            laps = [int(x) for x in re.findall('m-timestamp-([0-9]*)', msg)]
            if len(laps) >= 2:
                latency_us = laps[1] - laps[0]
                print(f"✅ Latency reported (from m-timestamp): {latency_us} us")
            else:
                laps = [int(x) for x in re.findall('m-lap-us-([0-9]*)', msg)]
                if len(laps) >= 2:
                    latency_us = laps[1] - laps[0]
                    print(f"✅ Latency reported (from m-lap-us): {latency_us} us")
                else:
                    print("⚠️ Could not parse latency from output.")
        except Exception as e:
            print(f"⚠️ Error parsing latency: {e}")

        # 8. Parse and Check Results
        print("Parsing results tensor...")
        m = re.search('m-results-\\[((?:-?[0-9]+,?)+)\\]', msg)
        
        if not m:
            print("\n" + "="*50)
            print("❌ FAILED: Could not find 'm-results-[]' in output!")
            print("="*50)
            print(f"Full message from board was:\n{msg}")
            raise Exception("Result parsing failed")
            
        board_results_str = m.group(1)
        board_results = [int(x) for x in board_results_str.split(',')]
        print(f"Board returned {len(board_results)} values.")
        
        if len(board_results) != OUTPUT_SIZE:
            print(f"❌ FAILED: Board returned wrong number of values! Expected {OUTPUT_SIZE}, got {len(board_results)}")
            raise Exception("Result size mismatch")
            
        # 9. Compare board vs. golden
        print("Comparing board results to golden file...")
        mismatch_count = 0
        first_mismatch_idx = -1
        for i in range(OUTPUT_SIZE):
            if board_results[i] != golden_results[i]:
                mismatch_count += 1
                if first_mismatch_idx == -1:
                    first_mismatch_idx = i
        
        if mismatch_count == 0:
            print("\n" + "="*50)
            print("✅ PASSED: All 4292 output values match the golden file!")
            print("✅ Your hardware and model are functionally correct.")
            print("="*50)
        else:
            print("\n" + "="*50)
            print(f"❌ FAILED: Found {mismatch_count} mismatches.")
            print(f"First mismatch at index {first_mismatch_idx}:")
            print(f"  Expected (golden): {golden_results[first_mismatch_idx]}")
            print(f"  Got (board):   {board_results[first_mismatch_idx]}")
            print("="*50)

    except Exception as e:
        print(f"\n--- An error occurred during the test ---")
        print(e)
    finally:
        com.close()
        print("Serial port closed.")

if __name__ == '__main__':
    main()