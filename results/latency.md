# LOFOP latency benchmark

Device `cpu`, 640px, AMP=False. Per-image latency in ms (lower is better); throughput in FPS.

| Variant | Batch | p50 (ms) | p90 (ms) | p99 (ms) | FPS |
|---|---:|---:|---:|---:|---:|
| lofop-detect-ex | 1 | 322.877 | 349.309 | 611.596 | 3.1 |
| lofop-detect-ex | 4 | 427.253 | 453.796 | 460.638 | 2.3 |
| lofop-detect-n | 1 | 44.532 | 50.611 | 54.41 | 22.5 |
| lofop-detect-n | 4 | 40.314 | 42.625 | 44.307 | 24.8 |
| lofop-detect-s | 1 | 86.629 | 93.645 | 99.862 | 11.5 |
| lofop-detect-s | 4 | 93.981 | 99.364 | 103.68 | 10.6 |
