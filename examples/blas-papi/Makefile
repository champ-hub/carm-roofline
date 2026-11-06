CC ?= cc
CFLAGS ?= -O2 -g -Wall -Wextra -pedantic -std=c11
LDLIBS ?= -lpapi -lm

.PHONY: all clean
all: blas_papi

blas_papi: blas_papi.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) $(PAPI_LIBS) -o $@ $< $(LDLIBS)

clean:
	rm -f blas_papi
