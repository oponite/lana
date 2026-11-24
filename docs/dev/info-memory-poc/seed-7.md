objects=256 len=4096 (96 KiB/object) seed=7 update_per_mille=10 repeats=3

| workload | store | steady heap MiB | peak heap MiB (fill) | ratio | exact | VmHWM MiB | alloc µs | compact µs | recall µs | compact total ms |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| random | conventional | 24.00 | 24.00 | 1.00 | yes | 26.80 | 0.0 | 0.0 | 25.6 | 0.0 |
| random | deflate | 22.97 | 23.53 | 1.04 | yes | 26.35 | 0.1 | 2312.9 | 320.1 | 601.3 |
| random | lana | 24.04 | 24.04 | 1.00 | yes | 26.81 | 11.4 | 347.0 | 24.7 | 90.9 |
| redundant | conventional | 24.00 | 24.00 | 1.00 | yes | 27.41 | 0.0 | 0.0 | 23.9 | 0.0 |
| redundant | deflate | 0.18 | 0.71 | 129.88 | yes | 4.19 | 0.0 | 139.5 | 31.2 | 37.2 |
| redundant | lana | 0.01 | 0.10 | 3576.72 | yes | 3.50 | 11.5 | 342.4 | 13.9 | 89.1 |
| structured | conventional | 24.00 | 24.00 | 1.00 | yes | 27.11 | 0.0 | 0.0 | 24.3 | 0.0 |
| structured | deflate | 22.97 | 23.53 | 1.04 | yes | 26.77 | 0.1 | 2240.8 | 318.3 | 576.6 |
| structured | lana | 0.96 | 1.06 | 24.93 | yes | 4.22 | 11.2 | 349.1 | 16.7 | 90.9 |

Reference (not a store): whole-store single DEFLATE stream, no random access

| workload | compressed MiB | ratio | total ms |
|---|---:|---:|---:|
| random | 22.96 | 1.05 | 655.8 |
| redundant | 0.16 | 146.93 | 32.5 |
| structured | 22.96 | 1.05 | 659.9 |
