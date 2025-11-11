#!/bin/bash

OUTPUT_FILE="time.txt"

> "$OUTPUT_FILE"

echo "Timing Examples (OpenACC) - $(date)" >> "$OUTPUT_FILE"
echo "======================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

time_example() {
    local example_name=$1
    local executable=$2
    
    if [ ! -f "$executable" ]; then
        echo "$example_name: EXECUTABLE NOT FOUND ($executable)" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        return
    fi
    
    echo "Running $example_name..." >&2
    
    start=$(date +%s%N)
    ./"$executable" > /dev/null 2>&1
    end=$(date +%s%N)
    
    elapsed=$(echo "scale=3; ($end - $start) / 1000000000" | bc)
    
    echo -n "$elapsed, " >> "$OUTPUT_FILE"
}

make
echo "OpenACC Versions:" >> "$OUTPUT_FILE"
echo "-------------" >> "$OUTPUT_FILE"

for (( i=0; i<5; i++ )); do
    time_example "Example 1 (OpenACC)" "example1_acc"
    time_example "Example 2 (OpenACC)" "example2_acc"
    time_example "Example 3 (OpenACC)" "example3_acc"
    time_example "Example 4 (OpenACC)" "example4_acc"
    time_example "Example 5 (OpenACC)" "example5_acc"
    echo "" >> "$OUTPUT_FILE"
done

echo "Timing complete! Results saved to $OUTPUT_FILE"
cat "$OUTPUT_FILE"
