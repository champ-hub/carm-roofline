COMPILER = g++
EXECUTABLE := Bench
FLAGS = -Wall -Wextra -O3 -std=c++17# should I do -O2 or O3?

BENCH:
	$(COMPILER) $(FLAGS) Bench.cpp create_bench.cpp -o $(EXECUTABLE) -Wno-unused-variable -Wno-unused-parameter

clean:
	rm -rf Bench ../bin