# CARM Examples

This directory contains example applications that demonstrate CARM integration and profiling workflows.

## Contents

### lulesh-papi

A [PAPI-instrumented fork](https://github.com/Alexandre425/LULESH-PAPI) of [LULESH](https://computing.llnl.gov/projects/co-design/lulesh) (Livermore Unstructured Lagrangian Explicit Shock Hydrodynamics), a proxy application developed by Lawrence Livermore National Laboratory. This fork adds PAPI hardware performance counter instrumentation for use with the CARM profiling pipeline.

**Purpose:** Demonstrates application-level profiling with CARM's PAPI-based performance measurement backend.

LULESH is licensed under the BSD 3-Clause License by LLNL; see the source headers (either the original or fork) for full license terms.

### topologies

TOML cache-topology examples for systems that do not expose cache metadata in sysfs.

`topologies/a64fx-48cpu.toml` describes a full 48-core Fujitsu A64FX node.
Use it with `carm benchmark --topology-config examples/topologies/a64fx-48cpu.toml`.
