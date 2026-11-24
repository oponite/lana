objects=256 len=4096 (96 KiB/object) seed=20261124 update_per_mille=100 repeats=3

| workload | store | steady heap MiB | peak heap MiB (fill) | ratio | exact | VmHWM MiB | alloc µs | compact µs | recall µs | compact total ms |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| random | conventional | 24.00 | 24.00 | 1.00 | yes | 26.73 | 0.0 | 0.0 | 23.1 | 0.0 |
| random | deflate | 22.97 | 23.53 | 1.04 | yes | 26.35 | 0.1 | 2312.5 | 321.5 | 596.3 |
| random | lana | 24.04 | 24.04 | 1.00 | yes | 26.80 | 11.5 | 346.5 | 27.4 | 89.8 |
| redundant | conventional | 24.00 | 24.00 | 1.00 | yes | 27.50 | 0.0 | 0.0 | 26.2 | 0.0 |
| redundant | deflate | 0.19 | 0.71 | 129.45 | yes | 4.19 | 0.1 | 147.2 | 32.8 | 41.0 |
| redundant | lana | 0.01 | 0.10 | 3576.72 | yes | 3.58 | 11.3 | 344.5 | 14.1 | 89.2 |
| structured | conventional | 24.00 | 24.00 | 1.00 | yes | 27.09 | 0.1 | 0.0 | 26.1 | 0.0 |
| structured | deflate | 22.98 | 23.54 | 1.04 | yes | 26.71 | 0.1 | 2252.4 | 315.3 | 578.1 |
| structured | lana | 5.42 | 5.52 | 4.43 | yes | 8.92 | 11.5 | 378.4 | 4.4 | 97.9 |

Reference (not a store): whole-store single DEFLATE stream, no random access

| workload | compressed MiB | ratio | total ms |
|---|---:|---:|---:|
| random | 22.96 | 1.05 | 661.2 |
| redundant | 0.16 | 146.84 | 33.4 |
| structured | 22.96 | 1.05 | 672.1 |
