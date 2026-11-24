objects=256 len=4096 (96 KiB/object) seed=20261124 update_per_mille=500 repeats=3

| workload | store | steady heap MiB | peak heap MiB (fill) | ratio | exact | VmHWM MiB | alloc µs | compact µs | recall µs | compact total ms |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| random | conventional | 24.00 | 24.00 | 1.00 | yes | 26.67 | 0.0 | 0.0 | 26.4 | 0.0 |
| random | deflate | 22.97 | 23.53 | 1.04 | yes | 26.35 | 0.1 | 2221.9 | 309.9 | 598.4 |
| random | lana | 24.04 | 24.04 | 1.00 | yes | 26.74 | 11.6 | 352.1 | 26.9 | 90.5 |
| redundant | conventional | 24.00 | 24.00 | 1.00 | yes | 27.48 | 0.0 | 0.0 | 23.9 | 0.0 |
| redundant | deflate | 0.19 | 0.71 | 129.45 | yes | 4.19 | 0.0 | 145.4 | 32.7 | 38.9 |
| redundant | lana | 0.01 | 0.10 | 3576.72 | yes | 3.54 | 11.0 | 335.1 | 13.7 | 86.0 |
| structured | conventional | 24.00 | 24.00 | 1.00 | yes | 27.03 | 0.1 | 0.0 | 26.9 | 0.0 |
| structured | deflate | 22.98 | 23.54 | 1.04 | yes | 26.69 | 0.1 | 2206.7 | 319.9 | 570.3 |
| structured | lana | 17.91 | 18.04 | 1.34 | yes | 22.09 | 11.4 | 412.0 | 11.3 | 106.3 |

Reference (not a store): whole-store single DEFLATE stream, no random access

| workload | compressed MiB | ratio | total ms |
|---|---:|---:|---:|
| random | 22.96 | 1.05 | 642.9 |
| redundant | 0.16 | 146.84 | 35.0 |
| structured | 22.96 | 1.05 | 669.1 |
