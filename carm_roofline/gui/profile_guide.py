"""Human-editable guide text for the profile configurator."""

from __future__ import annotations

PROFILE_GUIDE = """

## Basic usage

To profile your first application, choose **PAPI** if you have annotated it with PAPI region-of-interest calls
(PAPI HL), or **perf** for a simple time-slice based profiling. Both profilers run on the GUI server host.

Enter the executable and its arguments in **Command**, then select **Start profile**. The page shows progress, elapsed
time, output, and the generated command. Results are written on the GUI server.

For detailed instructions, see the `carm profile` [documentation](https://champ-hub.github.io/carm-roofline/profile.html).

## Parameters

### Backend
Choose **PAPI** for annotated-region data, or Linux **perf** otherwise. PAPI, when combined with "Region Merged"
aggregation, can produce one point per code region. perf will produce one point for the whole application, or per time
slice, if you specify a **perf interval** in the **Advanced settings**.

### Command
Enter the executable and arguments. Shell quoting keeps spaces in an argument, but commands are never run through a
shell. Put shell features in a script and profile that script.

### Aggregation
Choose how profiling results are grouped into roofline points:
- **Global** combines all ranks, threads, and regions into one point.
- **Rank** gives one point per rank.
- **Thread** gives one point per rank and thread.
- **Region merged** combines each region name across ranks and threads.
- **Region per thread** keeps a separate point for each region on each rank and thread.

Most of these aggregation modes are only available with PAPI.

### ISA
Optionally select one or more ISAs used by the workload to help select appropriate counters.
Leave unspecified if you do not know them.

### Data type
Optionally select the workload's data precision to improve estimates of operation and byte counts.
Leave unspecified if you do not know it.

## Advanced settings

You can hover over each setting for more information.
Machine and application names are optional labels.
The output directory is on the GUI server; its placeholder shows the default location.
Optional metrics are experimental and may not be accurate, or available on most systems.

If events exceed the hardware counter budget, **Run multiple times to collect all events** partitions collection and
merges the runs. This runs the application more than once, and can lead to inaccurate results.

PAPI caches its event catalog by machine configuration. Disable the **PAPI event cache** to refresh it or troubleshoot.
With **perf**, leave the interval empty for full-run mode, or enter a positive whole number of milliseconds for interval
sampling.

If you've made it this far, I recommend using the command-line interface. Run `carm profile --help` to see the available
options and what they do.
"""
