objects=256 len=4096 (96 KiB/object) seed=20261124 update_per_mille=900 repeats=3

| workload | store | steady heap MiB | peak heap MiB (fill) | ratio | exact | VmHWM MiB | alloc µs | compact µs | recall µs | compact total ms |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| random | conventional | 24.00 | 24.00 | 1.00 | yes | 26.67 | 0.0 | 0.0 | 24.6 | 0.0 |
| random | deflate | 22.97 | 23.53 | 1.04 | yes | 26.35 | 0.1 | 2209.7 | 305.0 | 569.3 |
| random | lana | 24.04 | 24.04 | 1.00 | yes | 26.82 | 11.0 | 335.5 | 24.6 | 85.9 |
| redundant | conventional | 24.00 | 24.00 | 1.00 | yes | 27.54 | 0.0 | 0.0 | 24.0 | 0.0 |
| redundant | deflate | 0.19 | 0.71 | 129.45 | yes | 4.19 | 0.0 | 138.2 | 31.1 | 36.5 |
| redundant | lana | 0.01 | 0.10 | 3576.72 | yes | 3.58 | 11.0 | 334.6 | 13.8 | 86.5 |
| structured | conventional | 24.00 | 24.00 | 1.00 | yes | 27.10 | 0.1 | 0.0 | 24.7 | 0.0 |
| structured | deflate | 22.98 | 23.54 | 1.04 | yes | 26.71 | 0.1 | 2244.1 | 316.9 | 576.7 |
| structured | lana | 23.60 | 23.72 | 1.02 | yes | 27.34 | 11.0 | 392.3 | 12.3 | 98.4 |

Reference (not a store): whole-store single DEFLATE stream, no random access

| workload | compressed MiB | ratio | total ms |
|---|---:|---:|---:|
| random | 22.96 | 1.05 | 674.5 |
| redundant | 0.16 | 146.84 | 32.7 |
| structured | 22.97 | 1.05 | 680.6 |
