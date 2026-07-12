# LOFOP latency benchmark

Device `cpu`, 640px, AMP=False. Per-image latency in ms (lower is better); throughput in FPS.

| Variant | Batch | p50 (ms) | p90 (ms) | p99 (ms) | FPS |
|---|---:|---:|---:|---:|---:|
| lofop-detect-ex | 1 | 320.215 | 391.286 | 472.091 | 3.1 |
| lofop-detect-ex | 4 | 404.798 | 434.6 | 441.743 | 2.5 |
| lofop-detect-n | 1 | 37.985 | 40.743 | 42.638 | 26.3 |
| lofop-detect-n | 4 | 32.602 | 35.917 | 37.313 | 30.7 |
| lofop-detect-s | 1 | 71.22 | 80.738 | 92.718 | 14.0 |
| lofop-detect-s | 4 | 82.09 | 90.892 | 92.541 | 12.2 |
