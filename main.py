"""RealPrice pipeline runner.

    python main.py                     # run everything (use after the raw data changed)
    python main.py --from-step training   # resume from a step
    python main.py --only eda          # run a single step
    python main.py --demo              # synthetic end-to-end smoke test (no real data needed)
"""
import argparse
import importlib
import os
import sys

STEPS = ["cleaning", "enrichment", "eda", "preprocessing", "training", "exporting", "build_website"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=STEPS, help="run just this step")
    parser.add_argument("--from-step", choices=STEPS, help="start from this step")
    parser.add_argument("--skip-eda", action="store_true", help="skip the (slowish) chart generation")
    parser.add_argument("--demo", action="store_true", help="generate synthetic data in demo_workspace/ and run on it")
    return parser.parse_args()


def select_steps(args: argparse.Namespace) -> list[str]:
    if args.only:
        return [args.only]
    steps = STEPS[STEPS.index(args.from_step):] if args.from_step else list(STEPS)
    return [s for s in steps if not (args.skip_eda and s == "eda")]


def main() -> None:
    args = parse_args()
    if args.demo:  # must happen before the config module is imported
        root = os.path.dirname(os.path.abspath(__file__))
        os.environ["REALPRICE_WORKSPACE"] = os.path.join(root, "demo_workspace")
        os.environ["REALPRICE_QUICK"] = "1"
    from realprice import config as cfg
    from realprice.utils import get_logger
    log = get_logger("main")
    if args.demo:
        from realprice.demo_data import generate
        generate()
    for name in select_steps(args):
        log.info("=" * 20 + f" step: {name} " + "=" * 20)
        importlib.import_module(f"realprice.{name}").run()
    log.info("done. workspace: %s", cfg.WORKSPACE)


if __name__ == "__main__":
    sys.exit(main())
