### Requirements for Codasip integration
- Option to 1) generate code only or 2) use a custom compiler and simulator (configuration file)
    - Option 1) would benefit from better test structure (memory and arith tests integrated in one binary)

### Technical notes
- x86 div requires additional

### Paraver Integration
- Add as many CARM metrics to paraver as possible
    - Add distance to the roof when plotting respective memory roof of each trace

### In-progress notes
- RISC-V VLEN detection needs to happen when the ISA is specified (if both RVV versions fail it skips it)

### Cool ideas
- Memory floor based on latency
    - Benchmark with something similar to pointer chase
- Integrate top-down metrics to associate app points to the respective roof, give confidence level
- Add a memory fragmentation metric
    - Based on ratio between L1 misses and bytes
    - 1 miss per 8 bytes -- high fragmentation, 1 miss per cache line size -- low fragmentation
    - Probably strongly correlated between position between floor and roof
    - Make synthetic benchmark to get 1 new cache line per instruction to verify if the performance is the same as the floor
- Support memory-only workloads by adding a vertical "one-dimensional" plot.
    - Horizontal lines show the bandwidth of each memory level, application bandwidth shows up as a point
