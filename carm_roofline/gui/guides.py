from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc

BENCHMARK_GUIDE = """

## Basic usage

To measure your first roofline, select the **Roofline** test, and leave the ISA detection as **Automatic detection**.
Select whatever data types and thread counts interest you, typically those matching an application you're analyzing. You
can als lower the **Run time** for faster benchmarking, 3s is usually good enough for a first test where you don't need
perfect accuracy.

Then click **Start benchmark**, the GUI shows progress, elapsed time, output, and the generated command.

For detailed instructions, see the `carm benchmark` [documentation](https://champ-hub.github.io/carm-roofline/benchmark.html).

## Parameters

### Test type
The type of benchmark to run. **Roofline** measures both arithmetic performance and memory bandwidth,
resulting in a full roof. You can also measure only arithmetic performance (**Arithmetic**) or only memory bandwidth
(**Memory**). **Mixed** validates the roofline by creating synthetic workloads at selected arithmetic intensities.

### ISA
The instruction set architecture to use for the benchmark. **Automatic detection** uses all ISAs the tool can detect on
the host. You can narrow it down to what interests you for faster benchmarking.

### Data types
The data types to use for the benchmark. You can select multiple data types in one run.

### Threads
The number of threads to use for the benchmark. You can select multiple thread counts in one run.

### Run time
The target duration of _each_ microbenchmark. Higher durations are more accurate, but take more time.

### Run name
The name of the given to the machine in the results files. This name is used to match it with corresponding apps, so use
the same name when profiling. Leave it blank and the tool automatically generates a name based on the CPU model.

## Advanced settings

You can hover over each of the advanced settings to read more about them.

If you've made it this far, I recommend using the command-line interface. Run `carm benchmark --help` to see all the
options available and what they do. It's very easy to use, and has less overhead than the GUI, meaning higher accuracy
for max-thread benchmarks.
"""


def build_guide_panel(title: str, markdown: str, visible: bool = True) -> dbc.Card:
    """Render human-editable Markdown in a shared, scrollable guide card."""
    return dbc.Card(
        [
            dbc.CardHeader(title, class_name="guide-panel-header"),
            dbc.CardBody(
                dcc.Markdown(markdown, className="guide-markdown", link_target="_blank"), class_name="guide-panel-body"
            ),
        ],
        class_name="guide-panel",
        style={"display": "block"} if visible else {"display": "none"},
    )
