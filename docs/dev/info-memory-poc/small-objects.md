objects=16384 len=64 (1 KiB/object) seed=20261124 update_per_mille=10 repeats=3

| workload | store | steady heap MiB | peak heap MiB (fill) | ratio | exact | VmHWM MiB | alloc µs | compact µs | recall µs | compact total ms |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| random | conventional | 24.25 | 24.25 | 0.99 | yes | 27.51 | 0.0 | 0.0 | 0.2 | 0.4 |
| random | deflate | 24.19 | 24.53 | 0.99 | yes | 27.85 | 0.0 | 26.4 | 10.9 | 446.1 |
| random | lana | 26.66 | 26.66 | 0.90 | yes | 29.91 | 0.3 | 5.9 | 0.3 | 100.3 |
| redundant | conventional | 24.25 | 24.25 | 0.99 | yes | 27.50 | 0.0 | 0.0 | 0.2 | 0.3 |
| redundant | deflate | 1.40 | 1.74 | 17.16 | yes | 5.09 | 0.0 | 10.0 | 4.7 | 169.3 |
| redundant | lana | 0.13 | 0.13 | 189.64 | yes | 3.18 | 0.2 | 5.5 | 0.1 | 92.5 |
| structured | conventional | 24.25 | 24.25 | 0.99 | yes | 27.55 | 0.0 | 0.0 | 0.2 | 0.3 |
| structured | deflate | 24.23 | 24.57 | 0.99 | yes | 27.89 | 0.0 | 25.8 | 10.4 | 434.7 |
| structured | lana | 3.53 | 4.00 | 6.80 | yes | 7.22 | 0.3 | 6.1 | 0.1 | 104.2 |

Reference (not a store): whole-store single DEFLATE stream, no random access

| workload | compressed MiB | ratio | total ms |
|---|---:|---:|---:|
| random | 22.96 | 1.05 | 673.0 |
| redundant | 0.08 | 309.03 | 37.8 |
| structured | 0.58 | 41.47 | 50.4 |
