# KLT (Kanade-Lucas-Tomasi) Feature Tracker with CUDA

This repository contains an implementation of the KLT feature tracker with both CPU and GPU (CUDA) versions.

## Building the Code

Navigate to the `src/v3` directory to build the code.

### CPU Version

To compile the CPU version:

```bash
cd src/v3
make cpu
```

This will:

1.  Compile the KLT CPU library (`libklt_cpu.a`).
2.  Build all CPU example programs (`example1_cpu` through `example5_cpu`).

### GPU Version

To compile the GPU version (requires CUDA toolkit):

```bash
cd src/v3
make gpu
```

This will:

1.  Compile the KLT GPU library (`libklt_gpu.a`).
2.  Build all GPU example programs (`example1_gpu` through `example5_gpu`).

## Running Examples

You can run each example individually from the `src/v3` directory.

### CPU Examples

```bash
cd src/v3

# Run CPU examples
./example1_cpu
./example2_cpu
# ... and so on for other examples
```

### GPU Examples

```bash
cd src/v3

# Run GPU examples
./example1_gpu
./example2_gpu
# ... and so on for other examples
```

## Cleaning Up

To clean compiled files:

```bash
cd src/v3
make clean
```
