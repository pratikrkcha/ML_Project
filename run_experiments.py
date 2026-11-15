"""CLI for the reproducible FedDef experiment matrix."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from itertools import product
from pathlib import Path

import pandas as pd

from src.config import Config
from src.data import load_data
from src.experiments import (
    DEFAULT_STRENGTHS,
    check_attack_strength,
    run_comparison,
    save_experiment_outputs,
)
from src.federated import run_federated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/framingham.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--matrix", choices=("core", "extended"), default="core")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--mal-fractions", type=float, nargs="+")
    parser.add_argument("--attack-types", choices=tuple(DEFAULT_STRENGTHS), nargs="+")
    parser.add_argument("--attack-strength", type=float, help="Override the default per-attack strength.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n-clients", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--theta", type=float, default=0.25)
    parser.add_argument("--z0", type=float, default=1.0)
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args(argv)

    fractions = args.mal_fractions or ([0.0, 0.1, 0.2, 0.3, 0.5] if args.matrix == "core" else [0.2])
    attacks = args.attack_types or (["random"] if args.matrix == "core" else list(DEFAULT_STRENGTHS))
    scenarios = list(product(dict.fromkeys(fractions), dict.fromkeys(attacks)))

    try:
        base = Config(
            seed=args.seeds[0], data_path=args.data, results_dir=args.results_dir,
            epochs=args.epochs, n_clients=args.n_clients, lr=args.lr, theta=args.theta, z0=args.z0,
        )
        configs = []
        for fraction, attack_type in scenarios:
            strength = DEFAULT_STRENGTHS[attack_type] if args.attack_strength is None else args.attack_strength
            check_attack_strength(attack_type, strength)
            configs.append(replace(
                base, mal_fraction=fraction, attack_type=attack_type, attack_strength=strength,
            ))
        if min(args.seeds) < 0:
            raise ValueError("Seeds must be non-negative")
    except ValueError as exc:
        parser.error(str(exc))

    include_test = not args.validation_only
    results = []
    for seed in dict.fromkeys(args.seeds):
        try:
            bundle = load_data(
                base.data_path, target_col=base.target_col, test_size=base.test_size,
                val_size=base.val_size, random_state=seed,
            )
        except (OSError, ValueError) as exc:
            parser.error(f"Could not load {base.data_path}: {exc}")

        baseline = run_federated(
            bundle, replace(base, seed=seed), evaluate_test=include_test, verbose=False,
        )
        for index, config in enumerate(configs):
            print(f"Seed {seed} | {config.attack_type} | malicious fraction {config.mal_fraction:.0%}", flush=True)
            metrics = run_comparison(
                bundle, replace(config, seed=seed), evaluate_test=include_test, clean_result=baseline,
            )
            if index > 0:
                metrics = metrics.loc[metrics["setting"].ne("clean")]
            results.append(metrics)

    metrics = pd.concat(results, ignore_index=True)
    save_experiment_outputs(metrics, args.results_dir)
    run_config = {
        "config": base.to_dict(),
        "seeds": list(dict.fromkeys(args.seeds)),
        "scenarios": [
            {"mal_fraction": case.mal_fraction, "attack_type": case.attack_type,
             "attack_strength": case.attack_strength}
            for case in configs
        ],
        "evaluate_test": include_test,
    }
    (args.results_dir / "experiment_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Saved {len(metrics):,} observations, aggregated metrics, and plots to {args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
