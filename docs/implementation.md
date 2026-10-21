# Implementation and verification record

User requirements: fixed video.mp4; six independent yolo11m.pt instances on each sampled camera frame; 15 analysis FPS per camera; portable Docker; 1/2/4/8/16 scaling; FPS, latency, GPU, VRAM, CPU, RAM; acceptance and maximum stable stream count.

Implementation boundaries:

- `config.py`: validated reproducible parameters, saved per run and stage.
- `pipeline.py`: independent time-paced looping readers, bounded latest-frame slots, fair scheduler, same-count warmup, complete-six-model frame accounting.
- `yolo.py`: six independent GPU models, actual-video preflight, synchronized CUDA timing.
- Version 1.1 adds `cuda_graph.py`: pre-captured six-model execution, shared input preparation, independent NMS and coordinate restoration, one explicit completion fence after all outputs. `--execution predict` preserves the original baseline. GPU output equivalence and profiling scripts are in `validate_optimization.py` and `diagnose.py`.
- `telemetry.py`: per-second whole-device resources, explicit unavailable values, CUDA VRAM fallback.
- `policy.py`: per-camera acceptance, exponential search with integer refinement, repeated long confirmation and reduction.
- `runner.py`: process isolation, timeouts, OOM recovery, software failure abort, environment and file hashes.
- `reporting.py`: stage traces, CSV/JSON and readable reports, explicit synthetic and upper-bound labels.
- Docker/Compose: pinned PyTorch/CUDA and direct Python dependencies; external video/weights/results mounts.

Verification sequence:

1. Write behavioral tests for acceptance, search, confirmation, frame accounting, missing telemetry and the software scheduler; observe absent-module failures before implementation.
2. Implement the corresponding modules and check all behaviors, including Windows timer granularity in the synthetic scheduler test.
3. Add subprocess CLI/report tests; implement stage isolation and CLI; verify synthetic results cannot claim hardware capacity.
4. Validate Compose configuration and build the image, running tests inside the build.
5. Download the fixed YOLO11-M checkpoint to the mounted host model directory.
6. Run real single-camera six-model smoke, inspect raw resources and frame latency.
7. Run the configured hardware capacity search and long confirmation when a candidate passes.

No historical measurements or unrelated files existed in the repository. The user's video is preserved and excluded from image context and Git by default.
