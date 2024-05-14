
all:
	cd Bench && make isa=${isa}
	
	#make -f Test/Makefile_Benchmark 
clean:
	cd Bench && make clean