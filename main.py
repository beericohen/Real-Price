"""Runs the RealPrice pipeline, step by step.

    python main.py                        # run everything (use after the raw data changed)
    python main.py --from-step training   # continue from a step (earlier results are reused)
    python main.py --only eda             # run a single step
    python main.py --skip-eda             # skip the charts (they take a while)
    python main.py --demo                 # test the whole pipeline on synthetic data (no real data needed)
"""
import argparse
import logging
import os
import time

# The order matters: each step reads the files written by the steps before it.
STEP_NAMES = ["cleaning", "enrichment", "eda", "preprocessing", "training", "exporting", "build_website"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=STEP_NAMES, help="run just this step")
    parser.add_argument("--from-step", choices=STEP_NAMES, help="start from this step")
    parser.add_argument("--skip-eda", action="store_true", help="skip the chart generation")
    parser.add_argument("--demo", action="store_true", help="generate synthetic data in demo_workspace/ and run on it")
    return parser.parse_args()


def choose_steps(args) -> list[str]:
    if args.only:
        return [args.only]
    steps = STEP_NAMES
    if args.from_step:
        steps = steps[steps.index(args.from_step):]
    if args.skip_eda:
        steps = [s for s in steps if s != "eda"]
    return steps


def main() -> None:
    args = parse_args()

    # config.py reads this variable when it is imported, so it must be set BEFORE the imports below.
    if args.demo:
        os.environ["REALPRICE_DEMO"] = "1"

    from realprice import build_website, cleaning, config, demo_data, eda, enrichment, exporting, preprocessing, training
    from realprice.utils import setup_logging

    setup_logging()
    log = logging.getLogger("main")
    run_functions = {
        "cleaning": cleaning.run,
        "enrichment": enrichment.run,
        "eda": eda.run,
        "preprocessing": preprocessing.run,
        "training": training.run,
        "exporting": exporting.run,
        "build_website": build_website.run,
    }

    if args.demo:
        demo_data.generate()

    for name in choose_steps(args):
        log.info("=" * 20 + f" step: {name} " + "=" * 20)
        start = time.perf_counter()
        run_functions[name]()
        log.info("step %s finished in %.1fs", name, time.perf_counter() - start)
    log.info("done. workspace: %s", config.WORKSPACE)


if __name__ == "__main__":
    main()
