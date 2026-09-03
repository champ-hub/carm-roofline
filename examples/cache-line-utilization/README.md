# Cache-line utilization

This serial PAPI HL example helps hit a target L1 cache-line utilization, its purpose being the testing and validation of the `carm profile` command, specifically the `--metric cache-line-utilization` option. You may also use it as reference for PAPI HL instrumentation.

Build it with `make`.

Run it with:

```sh
./cache_line_utilization CACHE_SIZE_KIB TARGET_CLU_PERCENT [PASSES]
```

For the given `CACHE_SIZE_KIB`, it will attempt to hit the target cache-line utilization percentage `TARGET_CLU_PERCENT`, repeating the experiment `PASSES` times (default is 100).
