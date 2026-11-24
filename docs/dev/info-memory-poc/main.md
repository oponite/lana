objects=1024 len=4096 (96 KiB/object) seed=20261124 update_per_mille=10 repeats=3

| workload | store | steady heap MiB | peak heap MiB (fill) | ratio | exact | VmHWM MiB | alloc µs | compact µs | recall µs | compact total ms |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| random | conventional | 96.02 | 96.02 | 1.00 | yes | 98.80 | 0.0 | 0.0 | 24.7 | 0.0 |
| random | deflate | 91.90 | 92.45 | 1.04 | yes | 95.36 | 0.1 | 2227.8 | 306.7 | 2342.4 |
| random | lana | 96.17 | 96.17 | 1.00 | yes | 99.07 | 11.5 | 342.6 | 26.9 | 354.6 |
| redundant | conventional | 96.02 | 96.02 | 1.00 | yes | 99.49 | 0.0 | 0.0 | 24.5 | 0.0 |
| redundant | deflate | 0.74 | 1.26 | 129.45 | yes | 4.79 | 0.1 | 143.6 | 32.6 | 151.3 |
| redundant | lana | 0.01 | 0.11 | 7637.58 | yes | 3.61 | 11.0 | 336.3 | 14.4 | 350.7 |
| structured | conventional | 96.02 | 96.02 | 1.00 | yes | 99.09 | 0.0 | 0.0 | 26.1 | 0.0 |
| structured | deflate | 91.91 | 92.47 | 1.04 | yes | 95.73 | 0.1 | 2323.6 | 316.3 | 2386.5 |
| structured | lana | 2.75 | 2.85 | 34.89 | yes | 6.23 | 11.2 | 350.7 | 16.4 | 371.4 |

Reference (not a store): whole-store single DEFLATE stream, no random access

| workload | compressed MiB | ratio | total ms |
|---|---:|---:|---:|
| random | 91.82 | 1.05 | 2767.3 |
| redundant | 0.65 | 146.88 | 130.7 |
| structured | 91.84 | 1.05 | 2619.6 |
