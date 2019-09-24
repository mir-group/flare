import bench_utils
import generate_kernel


"""
Script for generating benchmark data

Usage:

```bash
python generate.py --benchmark kernel --subtype two-body --location /tmp/
```

"""

args = bench_utils.parse_generate()

if args.benchmark == "kernel":
    generate_kernel.generate_kernel(args)
else:
    print("Benchmark type not found")
