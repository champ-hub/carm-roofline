# python run.py --test roofline --inst fma --plot --isa avx2

legacy_bench_gen/bench_sve -precision dp -test MEM -num_LD 2 -num_ST 2 -num_rep 1024
