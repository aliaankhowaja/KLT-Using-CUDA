# KLT (Kanade-Lucas-Tomasi) Feature Tracker with CUDA

This repository contains an implementation of the KLT feature tracker with both CPU and GPU (CUDA) versions, along with profiling capabilities.

## Building the Code

Navigate to the `src/v2` directory to build the code.

### CPU Version

To compile the CPU version:

```bash
cd src/v2
make cpu
```

This will:

1.  Compile the KLT CPU library (`libklt_cpu.a`).
2.  Build all CPU example programs (`example1_cpu` through `example5_cpu`).

### GPU Version

To compile the GPU version (requires CUDA toolkit):

```bash
cd src/v2
make gpu
```

This will:

1.  Compile the KLT GPU library (`libklt_gpu.a`).
2.  Build all GPU example programs (`example1_gpu` through `example5_gpu`).

## Running Examples

You can run each example individually from the `src/v2` directory.

### CPU Examples

```bash
cd src/v2

# Run CPU examples
./example1_cpu
./example2_cpu
# ... and so on for other examples
```

### GPU Examples

```bash
cd src/v2

# Run GPU examples
./example1_gpu
./example2_gpu
# ... and so on for other examples
```

## Profiling

TODO: Update for GPU

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
cd src/v2
make clean
```
