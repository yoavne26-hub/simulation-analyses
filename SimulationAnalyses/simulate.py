from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

from SeaWorldSimulation import SCENARIOS, run_single_scenario

DEFAULT_QUICK_SCENARIOS = ["BASE", "ALT1", "ALT2"]
DEFAULT_QUICK_RUNS = 5
DEFAULT_FULL_RUNS = 30


def parse_scenarios(raw: str) -> list[str]:
    if raw.strip().upper() == "ALL":
        return list(SCENARIOS.keys())
    parts = [p.strip().upper() for p in raw.split(",") if p.strip()]
    unknown = [p for p in parts if p not in SCENARIOS]
    if unknown:
        raise ValueError(f"Unknown scenarios: {', '.join(unknown)}")
    return parts


def build_rows(
    scenarios: Iterable[str],
    runs: int,
    seed: int,
) -> list[dict]:
    rows = []
    for scn_key in scenarios:
        scenario_name = SCENARIOS[scn_key]["label"]
        for run_id in range(runs):
            seed_used = seed + run_id
            result = run_single_scenario(scn_key, seed=seed_used)
            row = {
                "scenario_key": scn_key,
                "scenario_name": scenario_name,
                "run_id": run_id,
                "seed_used": seed_used,
            }
            row.update(result)
            rows.append(row)
    return rows


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(rows: list[dict], output_path: Path) -> None:
    ensure_parent_dir(output_path)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def summarize(rows: list[dict], output_path: Path) -> None:
    df = pd.DataFrame(rows)
    scenarios = ", ".join(sorted(df["scenario_key"].unique()))
    print("Simulation complete")
    print(f"  Scenarios: {scenarios}")
    print(f"  Runs per scenario: {df['run_id'].nunique()}")
    print(f"  Output: {output_path}")


def resolve_defaults(args: argparse.Namespace, has_args: bool) -> tuple[list[str], int]:
    if args.full:
        return list(SCENARIOS.keys()), DEFAULT_FULL_RUNS

    if args.quick or not has_args:
        scenarios = DEFAULT_QUICK_SCENARIOS if args.scenarios is None else parse_scenarios(args.scenarios)
        runs = DEFAULT_QUICK_RUNS if args.runs is None else args.runs
        return scenarios, runs

    scenarios = DEFAULT_QUICK_SCENARIOS if args.scenarios is None else parse_scenarios(args.scenarios)
    runs = DEFAULT_QUICK_RUNS if args.runs is None else args.runs
    return scenarios, runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SeaWorld simulation and write dataset CSV.")
    parser.add_argument("--scenarios", help="Comma-separated list (e.g., BASE,ALT1,ALT2) or ALL")
    parser.add_argument("--runs", type=int, help="Number of replications per scenario")
    parser.add_argument("--seed", type=int, default=102, help="Base random seed")
    parser.add_argument("--out", type=Path, default=Path("data/simulation_results.csv"), help="Output CSV path")
    parser.add_argument("--quick", action="store_true", help="Quick mode (default when no args)")
    parser.add_argument("--full", action="store_true", help="Full mode: ALL scenarios with 30 runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    has_args = len(sys.argv) > 1
    scenarios, runs = resolve_defaults(args, has_args)

    rows = build_rows(scenarios, runs, args.seed)
    write_csv(rows, args.out)
    summarize(rows, args.out)


if __name__ == "__main__":
    main()
