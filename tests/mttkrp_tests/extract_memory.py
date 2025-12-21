#!/usr/bin/env python3
"""
Simple ROCm Memory Operations Extractor
Extracts all allocation and free operations from rocprof JSON output
"""

import json
import sys


def extract_memory_operations(json_file, output_file='memory_operations.txt'):
    """Extract all hipMalloc and hipFree operations to a text file"""
    
    # Read JSON file
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {e}")
        sys.exit(1)
    
    # Get trace events
    if isinstance(data, dict):
        traces = data.get('traceEvents', [])
    elif isinstance(data, list):
        traces = data
    else:
        traces = []
    
    # Open output file
    with open(output_file, 'w') as out:
        out.write("Memory Operations from rocprof\n")
        out.write("=" * 70 + "\n\n")
        
        alloc_count = 0
        free_count = 0
        
        # Find all memory operations
        for event in traces:
            if not isinstance(event, dict):
                continue
            
            name = event.get('name', '')
            
            # Check if it's a memory operation
            if 'hipMalloc' in name or 'hipFree' in name:
                # Write the entire JSON entry
                out.write(json.dumps(event, indent=2))
                out.write("\n" + "-" * 70 + "\n\n")
                
                if 'hipMalloc' in name:
                    alloc_count += 1
                elif 'hipFree' in name:
                    free_count += 1
        
        # Summary at the end
        out.write("=" * 70 + "\n")
        out.write(f"Total Allocations: {alloc_count}\n")
        out.write(f"Total Frees: {free_count}\n")
    
    print(f"Extracted {alloc_count} allocations and {free_count} frees")
    print(f"Output written to: {output_file}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python extract_memory_ops.py <results.json> [output.txt]")
        print("\nExample:")
        print("  python extract_memory_ops.py results.json")
        print("  python extract_memory_ops.py results.json my_output.txt")
        sys.exit(1)
    
    json_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'memory_operations.txt'
    
    extract_memory_operations(json_file, output_file)