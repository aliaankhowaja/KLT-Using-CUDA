#!/bin/bash

OUTPUT_FILE="time.txt"

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
declare -a openacc_times

# Build CPU version
echo "Building CPU version..." >&2
make cpu >&2

# Build OpenACC version
echo "Building OpenACC version..." >&2
make openacc >&2

echo "" >&2
echo "Running benchmarks..." >&2

# Run 5 iterations
for (( i=0; i<5; i++ )); do
    echo "  Iteration $((i+1))/5..." >&2
    
    cpu_time=$(time_example "example3_cpu")
    cpu_times+=($cpu_time)
    
    openacc_time=$(time_example "example3_acc")
    openacc_times+=($openacc_time)
done

# Calculate averages
cpu_sum=0
openacc_sum=0

for time in "${cpu_times[@]}"; do
    cpu_sum=$(echo "$cpu_sum + $time" | bc)
done

for time in "${openacc_times[@]}"; do
    openacc_sum=$(echo "$openacc_sum + $time" | bc)
done

cpu_avg=$(echo "scale=3; $cpu_sum / 5" | bc)
openacc_avg=$(echo "scale=3; $openacc_sum / 5" | bc)

# Calculate speedup
if [ "$openacc_avg" != "0" ] && [ "$openacc_avg" != "-1" ]; then
    speedup=$(echo "scale=2; $cpu_avg / $openacc_avg" | bc)
else
    speedup="N/A"
fi

# Write results to file
{
    echo "KLT Example 3 Performance Comparison"
    echo "Timing Results - $(date)"
    echo "========================================"
    echo ""
    echo "┌────────────┬──────────────┬─────────────────┬─────────┐"
    echo "│ Iteration  │ CPU Time (s) │ OpenACC Time (s)│ Speedup │"
    echo "├────────────┼──────────────┼─────────────────┼─────────┤"
    
    for (( i=0; i<5; i++ )); do
        iter_speedup=$(echo "scale=2; ${cpu_times[$i]} / ${openacc_times[$i]}" | bc)
        printf "│ %-10s │ %12s │ %15s │ %7s │\n" "$((i+1))" "${cpu_times[$i]}" "${openacc_times[$i]}" "${iter_speedup}x"
    done
    
    echo "├────────────┼──────────────┼─────────────────┼─────────┤"
    printf "│ %-10s │ %12s │ %15s │ %7s │\n" "Average" "$cpu_avg" "$openacc_avg" "${speedup}x"
    echo "└────────────┴──────────────┴─────────────────┴─────────┘"
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
