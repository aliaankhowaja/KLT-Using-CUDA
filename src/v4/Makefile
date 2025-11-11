# v4 implementation with CPU and OpenACC versions

CC_CPU = gcc # GCC for CPU version
CC_ACC = nvc # NVIDIA HPC C compiler with OpenACC support

# OpenACC flags
ACCFLAGS = -acc -gpu=cc86 -Minfo=accel

CFLAGS_CPU = -O3 -Wno-unused-result
CFLAGS_ACC = $(ACCFLAGS) -O3

EXAMPLES = example1.c example2.c example3.c example4.c example5.c

# All source files for the library
SOURCES = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
          storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c

# Object files for CPU version
OBJECTS_CPU = $(SOURCES:.c=_cpu.o)

# Object files for OpenACC version
OBJECTS_ACC = $(SOURCES:.c=_acc.o)

LIB = -L/usr/local/lib -L/usr/lib

.PHONY: all cpu openacc clean

all: cpu

cpu: clean_cpu libklt_cpu.a $(EXAMPLES:.c=_cpu)

openacc: clean_acc libklt_acc.a $(EXAMPLES:.c=_acc)

# CPU build rules
%_cpu.o: %.c
	$(CC_CPU) -c $(CFLAGS_CPU) $< -o $@

libklt_cpu.a: $(OBJECTS_CPU)
	rm -f libklt_cpu.a
	ar ruv libklt_cpu.a $(OBJECTS_CPU)

example%_cpu: example%.c libklt_cpu.a
	$(CC_CPU) $(CFLAGS_CPU) -o $@ $< -L. -lklt_cpu $(LIB) -lm

# OpenACC build rules
%_acc.o: %.c
	$(CC_ACC) -c $(CFLAGS_ACC) $< -o $@

libklt_acc.a: $(OBJECTS_ACC)
	rm -f libklt_acc.a
	ar ruv libklt_acc.a $(OBJECTS_ACC)

example%_acc: example%.c libklt_acc.a
	$(CC_ACC) $(CFLAGS_ACC) -o $@ $< -L. -lklt_acc $(LIB) -lm

clean_cpu:
	rm -f *_cpu.o libklt_cpu.a *_cpu

clean_acc:
	rm -f *_acc.o libklt_acc.a *_acc

clean:
	rm -f *_cpu.o *_acc.o *.a *_cpu *_acc *.tar *.tar.gz \
	feat*.ppm features.ft features.txt feat1.txt feat2.txt feat2.fl ft2.txt ft3.txt
