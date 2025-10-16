#!/bin/bash

# Script to time each example and output results to time.txt

OUTPUT_FILE="time.txt"

# Clear the output file
> "$OUTPUT_FILE"

echo "Timing Examples - $(date)" >> "$OUTPUT_FILE"
echo "======================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Function to time an example
time_example() {
    local example_name=$1
    local executable=$2
    
    if [ ! -f "$executable" ]; then
        echo "$example_name: EXECUTABLE NOT FOUND ($executable)" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        return
    fi
    
    echo "Running $example_name..." >&2
    
    # Get start time in nanoseconds
    start=$(date +%s%N)
    
    # Run the executable
    ./"$executable" > /dev/null 2>&1
    
    # Get end time in nanoseconds
    end=$(date +%s%N)
    
    # Calculate elapsed time in seconds (with millisecond precision)
    elapsed=$(echo "scale=3; ($end - $start) / 1000000000" | bc)
    
    echo "$example_name: $elapsed seconds" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
}

# Time CPU versions
make cpu
echo "CPU Versions:" >> "$OUTPUT_FILE"
echo "-------------" >> "$OUTPUT_FILE"
time_example "Example 1 (CPU)" "example1_cpu"
time_example "Example 2 (CPU)" "example2_cpu"
time_example "Example 3 (CPU)" "example3_cpu"
time_example "Example 4 (CPU)" "example4_cpu"
time_example "Example 5 (CPU)" "example5_cpu"

make gpu
echo "" >> "$OUTPUT_FILE"
echo "GPU Versions:" >> "$OUTPUT_FILE"
echo "-------------" >> "$OUTPUT_FILE"
time_example "Example 1 (GPU)" "example1_gpu"
time_example "Example 2 (GPU)" "example2_gpu"
time_example "Example 3 (GPU)" "example3_gpu"
time_example "Example 4 (GPU)" "example4_gpu"
time_example "Example 5 (GPU)" "example5_gpu"

echo "Timing complete! Results saved to $OUTPUT_FILE"
cat "$OUTPUT_FILE"
