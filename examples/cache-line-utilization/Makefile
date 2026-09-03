CC ?= cc
CFLAGS ?= -O2 -g -Wall -Wextra -pedantic -std=c11 -D_POSIX_C_SOURCE=200112L
LDLIBS ?= -lpapi

all: cache_line_utilization

cache_line_utilization: cache_line_utilization.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f cache_line_utilization
