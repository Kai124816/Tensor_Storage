import csv
import argparse
from collections import defaultdict

def merge_multipass_rocprof_data_only(passes, output_csv_path, tensor_name, mode_val, counters):
    dispatches = defaultdict(dict)

    # The columns we care about (in order)
    target_headers = ["Tensor", "Mode"] + counters

    def process_file(file_path):
        try:
            with open(file_path, mode='r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    corr_id = row.get("Correlation_Id")
                    if not corr_id:
                        continue
                    counter_name = row.get("Counter_Name")
                    if counter_name in counters:
                        dispatches[corr_id][counter_name] = row.get("Counter_Value", "0")
        except FileNotFoundError:
            print(f"Warning: {file_path} not found.")

    # 1. Process passes
    for p in passes:
        process_file(p)

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
                    "Mode": mode_val
                }
                for c in counters:
                    row_to_write[c] = data.get(c, "0")

                writer.writerow(row_to_write)

        print(f"Data for {tensor_name} appended to {output_csv_path}")
    except Exception as e:
        print(f"Error writing output: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multipass rocprof data")
    parser.add_argument('--passes', nargs='+', required=True, help='Paths to the pass csv files (e.g., pass1.csv pass2.csv)')
    parser.add_argument('--counters-file', dest='counters_file', required=True, help='Path to .txt file containing list of counters to include')
    parser.add_argument('--output', required=True, help='Output CSV path')
    parser.add_argument('--tensor', required=True, help='Tensor Name')
    parser.add_argument('--mode', required=True, help='Mode')
    
    args = parser.parse_args()
    
    counters = []
    try:
        with open(args.counters_file, 'r') as cf:
            for line in cf:
                line = line.strip()
                if line and not line.startswith('#'):
                    counters.append(line)
    except Exception as e:
        import sys
        print(f"Error reading counters file: {e}")
        sys.exit(1)

    if len(args.passes) not in [1, 2, 3]:
        print("Warning: Expected 1, 2, or 3 passes, got", len(args.passes))
        
    merge_multipass_rocprof_data_only(args.passes, args.output, args.tensor, args.mode, counters)