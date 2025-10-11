CPU_DIR = ./cpu
GPU_DIR = ./gpu

CC = gcc # C compiler
NVCC = nvcc # NVIDIA CUDA compiler

CUDA_COMPUTE_CAPABILITY = compute_86 # Specify the compute capability of your GPU
CUDA_SM_ARCH = sm_86

CFLAGS = -pg -I$(CPU_DIR) # Compile with profiling information
NVCCFLAGS = -arch=$(CUDA_COMPUTE_CAPABILITY) -code=$(CUDA_SM_ARCH) -pg -I$(CPU_DIR) -I$(GPU_DIR)

EXAMPLES = example1.c example2.c example3.c example4.c example5.c

# C files in cpu directory
CPU_SOURCES = $(wildcard $(CPU_DIR)/*.c)
CPU_OBJECTS = $(patsubst $(CPU_DIR)/%.c,%.o,$(CPU_SOURCES))

# Common files needed for both CPU and GPU builds
COMMON_FILES_FOR_GPU = error.c pnmio.c pyramid.c selectGoodFeatures.c \
       storeFeatures.c klt.c klt_util.c writeFeatures.c
COMMON_OBJECTS_FOR_GPU = $(COMMON_FILES_FOR_GPU:.c=.o)

# CUDA files in the gpu directory
CUDA_SOURCES = $(wildcard $(GPU_DIR)/*.cu)
CUDA_OBJECTS = $(patsubst $(GPU_DIR)/%.cu,%.o,$(CUDA_SOURCES))

LIB = -L/usr/local/lib -L/usr/lib

.PHONY: all cpu gpu clean depend

all: cpu

cpu: libklt_cpu.a $(EXAMPLES:.c=_cpu)
gpu: libklt_gpu.a $(EXAMPLES:.c=_gpu)

# Rule to build .o from .c files in the cpu directory
%.o: $(CPU_DIR)/%.c
	$(CC) -c $(CFLAGS) $< -o $@

# Rule to build .o from .cu files in the gpu directory
%.o: $(GPU_DIR)/%.cu
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

libklt_cpu.a: $(CPU_OBJECTS)
	rm -f libklt_cpu.a
	ar ruv libklt_cpu.a $(CPU_OBJECTS)

# The GPU library depends on common objects (from cpu dir) and cuda objects (from gpu dir)
libklt_gpu.a: $(COMMON_OBJECTS_FOR_GPU) $(CUDA_OBJECTS)
	rm -f libklt_gpu.a
	ar ruv libklt_gpu.a $^

example%_cpu: example%.c libklt_cpu.a
	$(CC) -O3 $(CFLAGS) -o $@ $< -L. -lklt_cpu $(LIB) -lm

example%_gpu: example%.c libklt_gpu.a
	$(NVCC) -O3 $(NVCCFLAGS) -o $@ $< -L. -lklt_gpu $(LIB) -lm

depend:
	makedepend $(CPU_SOURCES) $(CUDA_SOURCES) $(EXAMPLES)

clean:
	rm -f *.o *.a *_cpu *_gpu *.tar *.tar.gz \
	feat*.ppm features.ft features.txt



