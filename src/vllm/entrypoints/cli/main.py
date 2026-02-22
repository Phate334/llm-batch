import argparse

from vllm.benchmarks import serve as bench_serve


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal vLLM CLI")
    subparsers = parser.add_subparsers(dest="subcmd")

    bench_parser = subparsers.add_parser("bench", help="Benchmark commands")
    bench_subparsers = bench_parser.add_subparsers(dest="bench_subcmd")

    serve_parser = bench_subparsers.add_parser(
        "serve", help="Benchmark the online serving throughput"
    )
    bench_serve.add_cli_args(serve_parser)
    serve_parser.set_defaults(dispatch_function=bench_serve.main)

    args = parser.parse_args()
    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
