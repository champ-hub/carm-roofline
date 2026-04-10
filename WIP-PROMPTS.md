/plan look at the #file:output  module, and how it handles plotting. I want to extract `plotting` to its own reusable module. It should use matplotlib to be able to plot the following:
- performance curve of high-granularity arithmetic / memory benchmarks (GOP/s per number of operations, and GB/s per array size), these aren't implemented yet but the functionality will be needed
- bar plots comparing the arithmetic performance and bandwidth of different ISAs (one bar per ISA per arithmetic, like the existing code, and one group of bars, one per memory level, per ISA for the memory)
- Cache-aware roofline model plots. Remember that these plot the performance roof (y axis, log) relative to the arithmetic intensity (x axis, log). Each roof (one per level) has a memory bound sloped section to the left (less ops per byte), with performance increasing until the ridge point is reached as the Arithmetic intensity (AI) grows. In the memory-bound section, performance equals the AI times the bandwidth. The ridge points are located at peak performance divided by the level's memory bandwidth. To the right there is the compute roof, where the performance equals the peak arithmetic performance.

A lot of metrics can be extracted from the methods of the underlying *BenchmarkSuite

come up with a design for the plotting module that allows it to be used independently (i.e. allow access to functions or methods that don't immediately save a plot, to allow the used to configure the plotting in detail). It should also integrate well with the existing output handlers

-------------
