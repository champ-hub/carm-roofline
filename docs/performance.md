---
title: Performance
parent: Paraver
nav_order: 4
---

# GUI Performance with Large Traces

A Paraver trace can contain millions of bursts. If the GUI feels slow with very large traces, try:

- **Narrow the time window** — drag the Time window slider handles to the range you care about. The displayed burst set shrinks.
- **Raise the minimum arithmetic intensity** — the leftmost slider position switches filtering off; a higher threshold hides low-intensity bursts.
- **Raise the minimum duration** — short bursts usually dominate the point count; hiding them removes many points.
