"""Single entry point for the LeWM toy experiments (see papers/leworldmodel.pdf).

Usage:
    python main.py                      # run all experiments in order
    python main.py <name>               # run one experiment

Experiments (each is self-contained and seeds its own RNG):
    param_recovery       teacher-forced L_pred recovers smooth linear dynamics
    circular_embedding   mod-100 dynamics: plain MSE stalls, circular MSE recovers
    sigreg_floor         a faithful periodic code carries an irreducible SIGReg floor
    penalty_comparison   SIGReg vs a circular penalty vs VICReg on a periodic factor
"""

import argparse

from experiments import EXPERIMENTS


def run(name):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    EXPERIMENTS[name]()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", nargs="?", default="all",
                        help="experiment to run, or 'all' (default): " + ", ".join(EXPERIMENTS))
    args = parser.parse_args()

    if args.experiment == "all":
        for name in EXPERIMENTS:
            run(name)
    elif args.experiment in EXPERIMENTS:
        run(args.experiment)
    else:
        parser.error(f"unknown experiment '{args.experiment}'. "
                     f"choose from: {', '.join(EXPERIMENTS)}, all")


if __name__ == "__main__":
    main()
