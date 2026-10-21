# Configured stream counts implementation plan

**Goal:** Explain every configuration field in Traditional Chinese and allow ascending, explicitly configured camera counts, stopping at the first failed count.

**Architecture:** Add an optional `test_streams` list to Config and ignore only the reserved `_help` dictionary when loading JSON. Keep legacy automatic search for null; explicit lists use a separate stop-on-failure policy, do not refine intermediate integers, and never claim an absolute maximum. Keep long confirmation of the highest passing listed count, stopping confirmation at its first failure without trying unlisted counts.

**Tech stack:** Python dataclasses/unittest, existing CUDA Graph pipeline, Docker Compose.

## Tasks

- [x] Add failing tests to `tests/test_config.py`: `_help` loading, unknown-key rejection, all configuration fields documented, valid ascending counts, rejection of empty/duplicate/decreasing/noninteger/nonpositive values, backward-compatible null.
- [x] Add failing policy tests: calls exactly `[1, 2, 4]` on failure at 4 from `[1, 2, 4, 8]`; first failure prevents all later counts; all-pass upper bound; confirmation failure does not fall back when disabled.
- [x] Add runner integration tests using real synthetic subprocesses: failing first configured stage stops at one stage; passing list confirms only its highest count; reports label configured scope, not absolute capacity.
- [x] Run `python -m unittest discover -s tests -v` to observe missing-feature failures; implement Config loader/validation, policy functions, runner routing and reporting. Preserve existing auto-search behavior and existing result files.
- [x] Add complete `_help`, `test_streams: [1, 2, 4, 8, 16]` and explicit `synthetic_delay_ms` to `config.json`; retain 10 FPS and all workload/SLA values. Update README for stop semantics, null/one-count examples and the actual 10 FPS results.
- [x] Run host tests, rebuild version 1.2 from the existing local dependency image, run all tests with CUDA, and perform one short two-camera real GPU smoke check. This is a functional check, not a new capacity confirmation.
- [x] Confirm Compose uses the new image and the annotated mounted configuration loads inside Docker. Remove the temporary local refresh Dockerfile. Leave all user files and original results intact.

Verification commands:

```powershell
python -m unittest discover -s tests -v
docker compose config --quiet
docker compose run --rm --entrypoint python benchmark -m unittest discover -s tests -v
docker compose run --rm benchmark smoke --streams 2 --duration 10
```

## Verification evidence

- Host: 38 tests successful, with 3 CUDA hardware tests explicitly skipped on the older host runtime.
- Docker GPU: all 38 tests executed and passed, including the three CUDA Graph hardware tests.
- Independent code review: no issues found; 24 configuration/policy/runner tests independently passed.
- Version 1.2 image: `sha256:d4c66cb2fc1cfb781897566d786e063f59a63f48a393bb4c5d3c9fa6576dc0ca`.
- GPU smoke: `results/20261021-091308-1f5627/summary.json`, `smoke_passed`, each stream 100/100 complete frames in 10 seconds, 10 FPS, no drops, all six models healthy. This is not a capacity search or long-duration confirmation.
- 28 runtime configuration fields have corresponding `_help` descriptions. Existing 10 FPS capacity results and original weights/video were preserved.
