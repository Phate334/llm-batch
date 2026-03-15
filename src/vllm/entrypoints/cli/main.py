import argparse
import importlib.metadata

from vllm.logger import init_logger

logger = init_logger(__name__)


def _get_version() -> str:
    for dist_name in ("vllm", "llm-batch"):
        try:
            return importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def main() -> None:
    # Lazy-load command modules to avoid eager imports in CLI startup.
    import vllm.entrypoints.cli.benchmark.main
    from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG, cli_env_setup

    cmd_modules = [vllm.entrypoints.cli.benchmark.main]

    cli_env_setup()

    parser = argparse.ArgumentParser(
        description="vLLM CLI",
        epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=_get_version(),
    )
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    cmds = {}
    for cmd_module in cmd_modules:
        new_cmds = cmd_module.cmd_init()
        for cmd in new_cmds:
            cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd

    args = parser.parse_args()
    if args.subparser in cmds:
        cmds[args.subparser].validate(args)

    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
