######################################################################
# Choose your favorite C compiler
CC = gcc
NVCC = nvcc

######################################################################
# Flags
FLAG1 = -DNDEBUG 
# FLAG2 = -DKLT_USE_QSORT
CFLAGS = $(FLAG1) $(FLAG2) -pg
LIBS = -lm -lcudart -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio

# Extra flags for profiling build with nvprof
PROFILE_FLAGS = -O2 -Wall
CUDA_FLAGS = -O2 -arch=sm_86

######################################################################
# Sources
EXAMPLES = example1.c example2.c example3.c example4.c example5.c
ARCH = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
       storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c
CUDA_ARCH = convolve.cu selectGoodFeatures.cu
LIB = -L/usr/local/lib -L/usr/lib -L/usr/local/cuda/lib64
LIB = libklt.a

.SUFFIXES:  .c .o

######################################################################
# Default build
all:  lib $(EXAMPLES:.c=)

.c.o:
	$(CC) -c $(CFLAGS) $<

lib: $(ARCH:.c=.o) convolve_cuda.o selectGoodFeatures_cuda.o
	rm -f libklt.a
	ar ruv libklt.a $(ARCH:.c=.o) convolve_cuda.o selectGoodFeatures_cuda.o
	rm -f *.o
	
convolve_cuda.o: $(CUDA_ARCH)
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@

selectGoodFeatures_cuda.o: selectGoodFeatures.cu
	$(NVCC) $(CUDA_FLAGS) -c selectGoodFeatures.cu -o $@

######################################################################
# Example builds (optimized, no profiling) - using nvcc for CUDA linking
example1: libklt.a
	$(NVCC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm -lcudart

example2: libklt.a
	$(NVCC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm -lcudart

example3: libklt.a
	$(NVCC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm -lcudart

example4: libklt.a
	$(NVCC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm -lcudart

example5: libklt.a
	$(NVCC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm -lcudart

######################################################################
# Profiling targets
profile_example1: libklt.a
	$(NVCC) $(CUDA_FLAGS) $(CFLAGS) -o example1 example1.c -L. -lklt $(LIB) -lm -lcudart
	nvprof ./example1 > profile_example1.txt 2>&1
	@echo "Profile written to profile_example1.txt"

profile_example2: libklt.a
	$(NVCC) $(CUDA_FLAGS) $(CFLAGS) -o example2 example2.c -L. -lklt $(LIB) -lm -lcudart
	nvprof ./example2 > profile_example2.txt 2>&1
	@echo "Profile written to profile_example2.txt"

profile_example3: libklt.a
	$(NVCC) $(CUDA_FLAGS) $(CFLAGS) -o example3 example3.c -L. -lklt $(LIB) -lm -lcudart
	nvprof ./example3 > profile_example3.txt 2>&1
	@echo "Profile written to profile_example3.txt"

profile_example4: libklt.a
	$(NVCC) $(CUDA_FLAGS) $(CFLAGS) -o example4 example4.c -L. -lklt $(LIB) -lm -lcudart
	nvprof ./example4 > profile_example4.txt 2>&1
	@echo "Profile written to profile_example4.txt"

profile_example5: libklt.a
	$(NVCC) $(CUDA_FLAGS) $(CFLAGS) -o example5 example5.c -L. -lklt $(LIB) -lm -lcudart 
	nvprof ./example5 ../../data/img0.pgm ../../data/img2.pgm > profile_example5.txt 2>&1
	gprof example5 gmon.out > profile_example5.txt
	@echo "Profile written to profile_example5.txt"

######################################################################
# Housekeeping
depend:
	makedepend $(ARCH) $(EXAMPLES)

clean:
	rm -f *.o *.a $(EXAMPLES:.c=) *.tar *.tar.gz libklt.a \
	      feat*.ppm features.ft features.txt gmon.out profile_*.txt
