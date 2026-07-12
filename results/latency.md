# LOFOP latency benchmark

Device `cpu`, 640px, AMP=False. Per-image latency in ms (lower is better); throughput in FPS.

| Variant | Batch | p50 (ms) | p90 (ms) | p99 (ms) | FPS |
|---|---:|---:|---:|---:|---:|
| lofop-detect-ex | 1 | 402.198 | 417.555 | 427.74 | 2.5 |
| lofop-detect-ex | 4 | 582.786 | 670.069 | 682.314 | 1.7 |
| lofop-detect-n | 1 | 46.601 | 50.926 | 54.29 | 21.5 |
| lofop-detect-n | 4 | 54.777 | 57.989 | 58.784 | 18.3 |
| lofop-detect-s | 1 | 92.455 | 100.119 | 103.059 | 10.8 |
| lofop-detect-s | 4 | 122.748 | 130.538 | 131.253 | 8.1 |
