# KLT (Kanade-Lucas-Tomasi) Feature Tracker

This repository contains an implementation of the KLT feature tracker in CUDA with profiling capabilities.

## Building the Code

Navigate to the source directory and compile:

```bash
cd src/v1
make all
```

This will:

1. Compile the KLT library (`libklt.a`)
2. Build all example programs (`example1` through `example5`)

## Running Examples

### Individual Examples

You can run each example individually:

```bash
cd src/v1

# Run example 1
./example1

# Run example 2
./example2

# Run example 3
./example3

# Run example 4
./example4

# Run example 5
./example5
```

## Profiling

To run all examples with profiling and generate performance reports:

```bash
cd src/v1
./profile_all.sh
```

This script will:

1. Compile all examples with profiling enabled
2. Run each example to collect performance data
3. Generate gprof reports
4. Create PDF visualizations of the profiling results

### Manual Profiling

To profile a single example manually:

```bash
cd src/v1
make all
./example1  # This generates gmon.out
gprof ./example1 gmon.out > example1.gprof
./gprof2pdf.sh example1.gprof  # Generates PDF report
```

## Cleaning Up

To clean compiled files:

```bash
cd src/v1
make clean
```
