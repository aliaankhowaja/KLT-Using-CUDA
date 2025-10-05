#!/bin/bash

# This script profiles all the example applications and generates PDF reports.

# Exit on any error
set -e

# Compile all the examples
echo "Compiling examples..."
make all

# List of examples to process
EXAMPLES="example1 example2 example3 example4 example5"

for example in $EXAMPLES
do
  echo "Processing $example..."

  # Run the example to generate gmon.out
  echo "Running $example..."
  if [ -f "$example" ]; then
    ./$example
  else
    echo "Error: $example not found. Make sure it was compiled."
    exit 1
  fi

  # Generate gprof output
  GPROF_OUTPUT="${example}.gprof"
  echo "Generating gprof output to ${GPROF_OUTPUT}..."
  gprof ./$example gmon.out > $GPROF_OUTPUT

  # Generate PDF from gprof output
  echo "Generating PDF for $example..."
  ./gprof2pdf.sh $GPROF_OUTPUT

  # Clean up intermediate files
  rm -f gmon.out $GPROF_OUTPUT ${example}.dot

  echo "$example processing complete. PDF created."
  echo "---------------------------------"
done

echo "All examples have been profiled."
