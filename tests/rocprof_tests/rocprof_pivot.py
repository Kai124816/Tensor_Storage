import csv
import re
import sys
from collections import defaultdict

def merge_multipass_rocprof_data_only(pass1_path, pass2_path, output_csv_path, tensor_name, mode_val):
    dispatches = defaultdict(dict)

    # The columns we care about (in order)
    target_headers = [
        "Tensor", "Mode", "SQ_INSTS_VALU", "SQ_INSTS_LDS", 
        "SerializedAtomicRatio", "VALUUtilization", "SIMD_UTILIZATION"
    ]

    def process_file(file_path):
        try:
            with open(file_path, mode='r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    corr_id = row["Correlation_Id"]
                    dispatches[corr_id][row["Counter_Name"]] = row["Counter_Value"]
        except FileNotFoundError:
            print(f"Warning: {file_path} not found.")

    # 1. Process passes
    process_file(pass1_path)
    process_file(pass2_path)

    if not dispatches:
        return

    # 2. Write the data rows (no header)
    try:
        # Use 'a' (append) or 'w' (write/overwrite) depending on your needs
        # Switching to 'a' is common if you are building a dataset row by row
        with open(output_csv_path, mode='a', newline='') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=target_headers, extrasaction='ignore')
            
            # REMOVED: writer.writeheader()

            for corr_id in sorted(dispatches.keys(), key=lambda x: int(x)):
                data = dispatches[corr_id]
                
                row_to_write = {
                    "Tensor": tensor_name,
                    "Mode": mode_val,
                    "SQ_INSTS_VALU": data.get("SQ_INSTS_VALU", "0"),
                    "SQ_INSTS_LDS": data.get("SQ_INSTS_LDS", "0"),
                    "SerializedAtomicRatio": data.get("SerializedAtomicRatio", "0"),
                    "VALUUtilization": data.get("VALUUtilization", "0"),
                    "SIMD_UTILIZATION": data.get("SIMD_UTILIZATION", "0")
                }
                writer.writerow(row_to_write)

        print(f"Data for {tensor_name} appended to {output_csv_path}")
    except Exception as e:
        print(f"Error writing output: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python script.py <pass1.csv> <pass2.csv> <output.csv> <TensorName> <Mode>")
    else:
        merge_multipass_rocprof_data_only(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])