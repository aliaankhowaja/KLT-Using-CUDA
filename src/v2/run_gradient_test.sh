#!/bin/bash
# Script to compile and run the gradient sum test

echo "=== Building Gradient Sum Test ==="
echo ""

# Compile the test program (trackFeatures.cu is included, so it's self-contained)
echo "Compiling test_gradient_sum.cu..."

nvcc -o test_gradient_sum test_gradient_sum.cu \
     -I./cpu -I./gpu -arch=compute_86 -code=sm_86

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Compilation failed!"
    echo ""
    exit 1
fi

echo "✓ test_gradient_sum compiled successfully"
echo ""
echo "=== Running Test ==="
echo ""

./test_gradient_sum

exit_code=$?
echo ""
if [ $exit_code -eq 0 ]; then
    echo "=== Test completed successfully ==="
else
    echo "=== Test failed with exit code $exit_code ==="
fi

exit $exit_code
