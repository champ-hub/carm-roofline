# Topology Examples

These TOML files provide cache hierarchies for systems that do not expose cache metadata in sysfs.

## A64FX full node

`a64fx-48cpu.toml` describes a full 48-core Fujitsu A64FX node with four 12-core cache-memory groups.

Use it with:

```bash
carm benchmark --topology-config examples/topologies/a64fx-48cpu.toml
```

Check the CPU allocation and visible memory before use. Edit the file for a partial-node allocation.
