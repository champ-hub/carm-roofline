# Test coverage

Prefer E2E tests at real CLI, API, and GUI boundaries for complex features.
Keep focused tests for unique invariants, edge cases, calculations, parsing, and failure paths.

## Pytest markers

Each test has exactly one primary marker. Add `native` only for tests that compile and execute real native benchmarks. Keep ISA availability checks as skip conditions.

| Marker | Meaning |
|---|---|
| `unit` | Focused, isolated algorithms, helpers, and model logic |
| `integration` | Cross-component behavior below a complete user workflow |
| `e2e` | Complete CLI, API, UI, or Paraver workflows |
| `native` | Orthogonal requirement for real native compiler and benchmark execution |

Run each category with:

```bash
pytest -m unit test/
pytest -m integration test/
pytest -m e2e test/
pytest -m "e2e and not native" test/
pytest -m native test/
```

## Paraver fixture data

The Paraver end-to-end scenario uses a pinned anonymized fixture release.
It downloads on first use into ignored `test/.cache/paraver/`.
Only tests requesting `paraver_fixture` download data.
Collection and other tests never access the network.
The downloader verifies the pinned SHA-256 before extracting the archive.
Unsafe archive entries are rejected.
Unavailable releases and checksum mismatches produce clear failures.

Prefetch the fixture before running Paraver tests offline, or before a CI test step:

```bash
python -m test.paraver_fixture
pytest test/test_e2e_paraver.py
```

The download contacts the public fixture release host and transfers the Paraver test archive. After prefetch, tests use the local cache without network access.


## E2E scenarios

Run the complete E2E scenario set with:

```bash
pytest test/test_e2e_cli_benchmark.py test/test_e2e_codegen.py test/test_e2e_cli_profile.py test/test_e2e_gui.py test/test_e2e_paraver.py
```

| Feature | E2E file | Coverage and isolated external work |
|---|---|---|
| Benchmark CLI | `test_e2e_cli_benchmark.py` | Tests arithmetic, topology memory, ordered memory sweeps, roofline, compiler/simulator, and dry-run artifacts. Checks bandwidth, JSONL/CSV/table, decoded nonuniform PNGs, and no dry-run binaries. Fakes compilation and measurements. |
| Native code generation | `test_e2e_codegen.py` | Runs arithmetic, memory, mixed, and roofline on x86-64 with GCC. AVX2 runs when detected. Cross-ISA generation remains focused coverage. |
| Profiling CLI | `test_e2e_cli_profile.py` | Checks real CLI workflows, rank-file and perf CSV parsing, aggregation, merges, metrics, and errors. Fixtures or mocks replace external apps, discovery, counters, and `perf`. |
| GUI | `test_e2e_gui.py` | Tests Dash HTTP routes and callbacks for results, settings, plots, and benchmark refresh. Mocks only benchmark execution. |
| Paraver | `test_e2e_paraver.py` | Uses a pinned anonymized trace and real `paramedir`-derived counters from the local cache. Tests GUI callbacks, filters, colors, and exports; mocks `paramedir` execution. |

GUI and Paraver use Dash's HTTP client, not a browser. Tests check layouts, callback responses, plots, and files.

## Focused coverage

- Keep focused tests for behavior E2E does not cover: ISA generation parameters and loop limits; deduplication identity and result propagation.
- Retain malformed-input, sparse-aggregation, metric-formula, fallback, backend-error, timeout, and optional-dependency tests.
- `test_dynamorio_client.py` builds and runs the real DynamoRIO client to verify first-instruction region boundaries, nested exclusive instruction/memory attribution, marker exclusion, and FMA3/FMA4 classification. FMA4 bytes are decoded and classified by DynamoRIO without execution, so this portion does not require FMA4 host support; the smoke app itself requires x86-64 AVX/FMA and a C/C++ toolchain.
- Assert observable behavior, not wording, source layout, or mock-only argument forwarding.

E2E does not collect real PAPI hardware counters or run MPI/OpenMP applications. Paraver's `paramedir` execution is mocked.
Native benchmark execution requires x86-64 and GCC. Focused ISA tests check generation invariants, not cross-target compilation or execution.
