

CC = gcc

FLAG1 = -DNDEBUG
# FLAG2 = -DKLT_USE_QSORT
CFLAGS = $(FLAG1) $(FLAG2)

# Extra flags for profiling build
PROFILE_FLAGS = -pg -O2 -Wall

# Sources
EXAMPLES = example1.c example2.c example3.c example4.c example5.c
ARCH = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
       storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c
LIB = -L/usr/local/lib -L/usr/lib

.SUFFIXES:  .c .o

# Default build
all:  lib $(EXAMPLES:.c=)

.c.o:
	$(CC) -c $(CFLAGS) $<

lib: $(ARCH:.c=.o)
	rm -f libklt.a
	ar ruv libklt.a $(ARCH:.c=.o)
	rm -f *.o

# Example builds (optimized, no profiling)
example1: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

example2: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

example3: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

example4: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

example5: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

# Profiling targets

prof_ex1: libklt.a
	$(CC) $(PROFILE_FLAGS) $(CFLAGS) -o example1 example1.c -L. -lklt $(LIB) -lm
	./example1 ../../data/img0.pgm ../../data/img1.pgm
	gprof example1 gmon.out > prof_ex1.txt
	@echo "Profile written to prof_ex1.txt"

prof_ex2: libklt.a
	$(CC) $(PROFILE_FLAGS) $(CFLAGS) -o example2 example2.c -L. -lklt $(LIB) -lm
	./example2 ../../data/img0.pgm ../../data/img1.pgm
	gprof example2 gmon.out > prof_ex2.txt
	@echo "Profile written to prof_ex2.txt"

prof_ex3: libklt.a
	$(CC) $(PROFILE_FLAGS) $(CFLAGS) -o example3 example3.c -L. -lklt $(LIB) -lm
	./example3 ../../data/img0.pgm ../../data/img1.pgm
	gprof example3 gmon.out > prof_ex3.txt
	@echo "Profile written to prof_ex3.txt"

prof_ex4: libklt.a
	# example4 depends on features.txt produced by example3
	@if [ ! -f features.txt ]; then \
		echo "Generating features.txt using example3..."; \
		$(CC) $(PROFILE_FLAGS) $(CFLAGS) -o example3 example3.c -L. -lklt $(LIB) -lm; \
		./example3 ../../data/img0.pgm ../../data/img1.pgm; \
	fi
	$(CC) $(PROFILE_FLAGS) $(CFLAGS) -o example4 example4.c -L. -lklt $(LIB) -lm
	./example4
	gprof example4 gmon.out > prof_ex4.txt
	@echo "Profile written to prof_ex4.txt"

prof_ex5: libklt.a
	$(CC) $(PROFILE_FLAGS) $(CFLAGS) -o example5 example5.c -L. -lklt $(LIB) -lm
	./example5 ../../data/img0.pgm ../../data/img1.pgm
	gprof example5 gmon.out > prof_ex5.txt
	@echo "Profile written to prof_ex5.txt"

######################################################################
# Run all profiling at once
######################################################################
profile_all: clean lib prof_ex1 prof_ex2 prof_ex3 prof_ex4 prof_ex5
	@mkdir -p profiles
	@mv prof_ex*.txt profiles/
	@echo "All profiling complete. Results saved in profiles/ folder."

# Housekeeping
depend:
	makedepend $(ARCH) $(EXAMPLES)

clean:
	rm -f *.o *.a $(EXAMPLES:.c=) *.tar *.tar.gz libklt.a \
	      feat*.ppm features.ft features.txt gmon.out profile_*.txt
