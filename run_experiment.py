from __future__ import annotations

import argparse
from pathlib import Path

from lap_associative_memory import ExperimentConfig, plot_results, run_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LAP associative-memory experiment")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="small end-to-end smoke experiment (default)")
    mode.add_argument("--full", action="store_true", help="multi-seed research scan")
    parser.add_argument("--output", type=Path, default=Path("results/quick"))
    parser.add_argument("--device", default=None, help="cpu, cuda, or mps; defaults to CUDA when available")
    args = parser.parse_args()

    config = ExperimentConfig.full() if args.full else ExperimentConfig()
    metrics, history = run_grid(config, args.output, device=args.device)
    figures = plot_results(metrics, history, args.output)
    print(f"Saved {len(metrics)} metric rows and {len(history)} history rows to {args.output}")
    for figure in figures:
        print(figure)


if __name__ == "__main__":
    main()
