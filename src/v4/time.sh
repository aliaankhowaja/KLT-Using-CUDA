#!/bin/bash

OUTPUT_FILE="time.txt"
NUM_ITERATIONS=1

> "$OUTPUT_FILE"

time_example() {
    local executable=$1
    
    if [ ! -f "$executable" ]; then
        echo "-1"
        return
    fi
    
    start=$(date +%s%N)
    ./"$executable" > /dev/null 2>&1
    end=$(date +%s%N)
    
    elapsed=$(echo "scale=3; ($end - $start) / 1000000000" | bc)
    echo "$elapsed"
}

# Arrays to store timing results
declare -a cpu_times
declare -a gpu_times

# Build CPU version
echo "Building CPU version..." >&2
make cpu >&2

# Build OpenACC version
echo "Building OpenACC version..." >&2
make openacc >&2

echo "" >&2
echo "Running benchmarks..." >&2

# Run NUM_ITERATIONS iterations
for (( i=0; i<NUM_ITERATIONS; i++ )); do
    echo "  Iteration $((i+1))/$NUM_ITERATIONS..." >&2
    
    echo "Running CPU build..."
    cpu_time=$(time_example "example3_cpu")
    cpu_times+=($cpu_time)
    
    echo "Running OpenACC build"
    gpu_time=$(time_example "example3_acc")
    gpu_times+=($gpu_time)
done

# Calculate averages
cpu_sum=0
gpu_sum=0

for time in "${cpu_times[@]}"; do
    cpu_sum=$(echo "$cpu_sum + $time" | bc)
done

for time in "${gpu_times[@]}"; do
    gpu_sum=$(echo "$gpu_sum + $time" | bc)
done

cpu_avg=$(echo "scale=3; $cpu_sum / $NUM_ITERATIONS" | bc)
gpu_avg=$(echo "scale=3; $gpu_sum / $NUM_ITERATIONS" | bc)

# Calculate speedup
if [ "$gpu_avg" != "0" ] && [ "$gpu_avg" != "-1" ]; then
    speedup=$(echo "scale=2; $cpu_avg / $gpu_avg" | bc)
else
    speedup="N/A"
fi

# Write results to file
{
    echo "KLT Example 3 Performance Comparison (OpenACC)"
    echo "Timing Results - $(date)"
    echo "========================================"
    echo ""
    echo "┌────────────┬──────────────┬──────────────┬─────────┐"
    echo "│ Iteration  │ CPU Time (s) │  OpenAcc (s) │ Speedup │"
    echo "├────────────┼──────────────┼──────────────┼─────────┤"
    
    for (( i=0; i<NUM_ITERATIONS; i++ )); do
        iter_speedup=$(echo "scale=2; ${cpu_times[$i]} / ${gpu_times[$i]}" | bc)
        printf "│ %-10s │ %12s │ %12s │ %7s │\n" "$((i+1))" "${cpu_times[$i]}" "${gpu_times[$i]}" "${iter_speedup}x"
    done
    
    echo "├────────────┼──────────────┼──────────────┼─────────┤"
    printf "│ %-10s │ %12s │ %12s │ %7s │\n" "Average" "$cpu_avg" "$gpu_avg" "${speedup}x"
    echo "└────────────┴──────────────┴──────────────┴─────────┘"
    echo ""
    
    if [ "$speedup" != "N/A" ]; then
        improvement=$(echo "scale=1; ($speedup - 1) * 100" | bc)
        echo "Performance Summary:"
        echo "  • OpenACC version is ${speedup}x faster than CPU"
        echo "  • Performance improvement: ${improvement}%"
    fi
    
} | tee "$OUTPUT_FILE"

echo ""
echo "Results saved to $OUTPUT_FILE"
