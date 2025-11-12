# v4 implementation with CPU and OpenACC versions

CPU_DIR = ./cpu
GPU_DIR = ./gpu

CC_CPU = gcc # GCC for CPU version
CC_ACC = nvc # NVIDIA HPC C compiler with OpenACC support
NVCC = nvcc # NVIDIA CUDA compiler

# OpenACC flags
ACCFLAGS = -acc -gpu=cc86 -Minfo=accel

CUDA_COMPUTE_CAPABILITY = compute_86 # compute capability
CUDA_SM_ARCH = sm_86 # specific architecture

CFLAGS_CPU = -Wno-unused-result
CFLAGS_ACC = $(ACCFLAGS)
CFLAGS = -I$(CPU_DIR)
NVCCFLAGS = -arch=$(CUDA_COMPUTE_CAPABILITY) -code=$(CUDA_SM_ARCH) -I$(CPU_DIR) -I$(GPU_DIR)

EXAMPLES = example1.c example2.c example3.c example4.c example5.c

# All source files for the library
SOURCES = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
          storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c

# Object files for CPU version
OBJECTS_CPU = $(SOURCES:.c=_cpu.o)

# Object files for OpenACC version
OBJECTS_ACC = $(SOURCES:.c=_acc.o)

# Common files needed for GPU builds (from CPU directory if using v3 structure)
COMMON_FILES_FOR_GPU = error.c pnmio.c storeFeatures.c klt_util.c writeFeatures.c
COMMON_OBJECTS_FOR_GPU = $(addprefix $(CPU_DIR)/, $(COMMON_FILES_FOR_GPU:.c=.o))

# CUDA files in the gpu directory
CUDA_SOURCES = $(wildcard $(GPU_DIR)/*.cu)
CUDA_OBJECTS = $(patsubst $(GPU_DIR)/%.cu,$(GPU_DIR)/%.o,$(CUDA_SOURCES))

LIB = -L/usr/local/lib -L/usr/lib

.PHONY: all cpu openacc gpu clean clean_cpu clean_acc clean_gpu

all: cpu

cpu: clean_cpu libklt_cpu.a $(EXAMPLES:.c=_cpu)

openacc: clean_acc libklt_acc.a $(EXAMPLES:.c=_acc) rm -f feat*.ppm

gpu: clean_gpu libklt_gpu.a $(EXAMPLES:.c=_gpu)

# CPU build rules
%_cpu.o: %.c
	$(CC_CPU) -O3 -c $(CFLAGS_CPU) $< -o $@

libklt_cpu.a: $(OBJECTS_CPU)
	rm -f libklt_cpu.a
	ar ruv libklt_cpu.a $(OBJECTS_CPU)

example%_cpu: example%.c libklt_cpu.a
	$(CC_CPU) -O3 $(CFLAGS_CPU) -o $@ $< -L. -lklt_cpu $(LIB) -lm

# OpenACC build rules
%_acc.o: %.c
	$(CC_ACC) -c $(CFLAGS_ACC) $< -o $@

libklt_acc.a: $(OBJECTS_ACC)
	rm -f libklt_acc.a
	ar ruv libklt_acc.a $(OBJECTS_ACC)

example%_acc: example%.c libklt_acc.a
	$(CC_ACC) -O3 $(CFLAGS_ACC) -o $@ $< -L. -lklt_acc $(LIB) -lm

# GPU build rules (from v3)
$(GPU_DIR)/%.o: $(GPU_DIR)/%.cu
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

$(CPU_DIR)/%.o: $(CPU_DIR)/%.c
	$(CC_CPU) -c $(CFLAGS) $< -o $@

libklt_gpu.a: $(COMMON_OBJECTS_FOR_GPU) $(CUDA_OBJECTS)
	rm -f libklt_gpu.a
	ar ruv libklt_gpu.a $^

example%_gpu: example%.c libklt_gpu.a
	$(NVCC) -O3 $(NVCCFLAGS) -o $@ $< -L. -lklt_gpu $(LIB) -lm

clean_cpu:
	rm -f *_cpu.o libklt_cpu.a *_cpu

clean_acc:
	rm -f *_acc.o libklt_acc.a *_acc

clean_gpu:
	rm -f $(GPU_DIR)/*.o $(CPU_DIR)/*.o libklt_gpu.a *_gpu

clean:
	rm -f *_cpu.o *_acc.o $(CPU_DIR)/*.o $(GPU_DIR)/*.o *.a *_cpu *_acc *_gpu *.tar *.tar.gz \
	feat*.ppm features.ft features.txt feat1.txt feat2.txt feat2.fl ft2.txt ft3.txt
