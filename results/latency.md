# LOFOP latency benchmark

Device `cpu`, 640px, AMP=False. Per-image latency in ms (lower is better); throughput in FPS.

| Variant | Batch | p50 (ms) | p90 (ms) | p99 (ms) | FPS |
|---|---:|---:|---:|---:|---:|
| lofop-detect-ex | 1 | 317.026 | 336.827 | 342.55 | 3.2 |
| lofop-detect-ex | 4 | 330.469 | 355.48 | 362.877 | 3.0 |
| lofop-detect-n | 1 | 43.222 | 45.676 | 46.759 | 23.1 |
| lofop-detect-n | 4 | 29.624 | 34.291 | 35.797 | 33.8 |
| lofop-detect-s | 1 | 83.594 | 90.789 | 96.3 | 12.0 |
| lofop-detect-s | 4 | 71.993 | 76.779 | 80.796 | 13.9 |
