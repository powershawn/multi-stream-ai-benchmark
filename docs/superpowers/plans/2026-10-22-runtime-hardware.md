# Runtime hardware reporting implementation plan

**Goal:** Every new run automatically records its actual GPU, runtime CPU/RAM and software; Windows launcher supplies a fresh, separately scoped physical-host snapshot.

**Architecture:** `hardware.py` collects CPU/RAM without initializing CUDA in the parent process. The existing GPU worker reports its CUDA-selected device properties, UUID and driver; runner merges these into the current run's environment. `run.ps1` passes a freshly collected CIM snapshot as a per-command base64 environment variable, never as a reused file. Report/CSV render saved metadata only, never inspect the report viewer's current machine.

**Tech stack:** Python 3.9+, psutil, existing Torch/NVML, PowerShell CIM, Docker Compose.

## Approved behavior

- GPU is the CUDA device actually selected by `gpu_device`, not blindly the first nvidia-smi row. Runtime capture must not add a parent CUDA context to benchmark memory usage.
- CPU model, physical/logical CPU counts, available affinity and runtime-visible RAM are scoped to the benchmark environment. Docker/WSL-visible RAM is never called physical Windows RAM.
- Windows snapshot contains collection time, OS, CPU and installed/visible RAM. Invalid, missing or stale snapshots are explicitly absent, not reused. Snapshot is collected after image build/model preparation and supplied only to the one benchmark invocation.
- New report.md and environment.json/summary.json include hardware/software; summary.csv includes machine identifiers for comparison. Old reports/results are not retroactively regenerated or changed.
- No change to six-model inference, 10 FPS, routes, timing or acceptance. No new full capacity claim from functional smoke checks.

## Steps

- [x] Add failing hardware tests: CPU model parsing, non-first selected GPU properties, fresh/missing/stale/malformed host snapshots, real runtime CPU/RAM, Windows CIM script output.
- [x] Add failing report/integration tests: two saved hardware fixtures render different machines; legacy/missing fields stay unknown; selected GPU propagates to the saved environment; CSV includes comparison fields.
- [x] Implement `benchmark/hardware.py`, integrate existing GPU worker and reporting. Run `python -m unittest discover -s tests -v` to green. No changes to YoloModels inference were necessary.
- [x] Implement `scripts/Get-HostInfo.ps1` and per-command injection in `run.ps1`, add `-NoBuild` for existing imported/local images. Check syntax and the real Windows collector without starting a benchmark.
- [x] Update README and image/version to 1.3. Copy scripts into Docker for the collector-presence test; it is executed only on the Windows host. Retain portable primary Dockerfile; build locally using installed 1.2 dependency layer.
- [x] Run all CUDA container tests; run short real GPU tests directly via Docker and via Windows launcher, checking actual GPU selection and absent/present host snapshots. Do not rerun capacity search.
- [x] Independent code review, fixes with regression tests, fresh final verification and cleanup of temporary refresh Dockerfile. Do not commit/push without a new user request.

Verification commands:

```powershell
python -m unittest discover -s tests -v
./scripts/Get-HostInfo.ps1 -AsBase64
docker compose config --quiet
docker compose run --rm --entrypoint python benchmark -m unittest discover -s tests -v
docker compose run --rm benchmark smoke --streams 1 --duration 10
./run.ps1 -Mode smoke -NoBuild
```

## Verification evidence

- Native Windows: 55 tests, no failures, 3 CUDA Graph tests explicitly skipped on the older host Torch. Real CIM collector executed and validated.
- Docker with GPU: 55 tests, no failures, only the Windows-only CIM test skipped. All CUDA Graph tests executed.
- Docker build without GPU: 55 tests, no failures, 4 expected platform/hardware skips.
- Independent review: no remaining findings; all 17 hardware/reporting tests independently passed after malformed nested JSON hardening and non-first-GPU propagation coverage.
- Direct Docker smoke: `results/20261022-055327-b20ab4/`, 10 seconds at 10 FPS, six models, passed; runtime hardware auto-recorded and `windows_host=null`.
- Windows launcher smoke: `results/20261022-055629-86509b/`, 15 seconds at 10 FPS, six models, passed; actual GPU UUID matches worker, physical host RAM 32 GiB and runtime-visible RAM 15.47 GiB rendered separately. Host snapshot supplied only to this invocation.
- Final image 1.3: `sha256:f7ef9207030befdbd6ee648d86d73f12c9f0b4f7d9add182d6e8ed73d13a3b3c`.
- Existing config exactly matches the preceding full capacity run (including 10 FPS/SLA/six weights). No historical result files regenerated, no new capacity assertion from smoke tests, no commit/push performed.
