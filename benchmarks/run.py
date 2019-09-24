import bench_utils
import run_kernel


args = bench_utils.parse_generate()

if args.benchmark == "kernel":
    run_kernel.run_kernel(args)
else:
    print("Benchmark type not found")
