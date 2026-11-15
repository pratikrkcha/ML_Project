from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from run_experiments import main
from src.config import Config
from src.data import prepare_data
from src.experiments import METRICS, run_comparison


def test_comparison_pairs_malicious_clients_and_keeps_test_held_out(synthetic_framingham):
    config = Config(seed=42, n_clients=5, epochs=2, mal_fraction=0.4, attack_type="label_flip")
    bundle = prepare_data(synthetic_framingham, random_state=config.seed)
    comparison = run_comparison(bundle, config, evaluate_test=False)
    pd.testing.assert_frame_equal(comparison, run_comparison(bundle, config, evaluate_test=False))

    seeded_ids = np.random.default_rng([config.seed, 0]).permutation(config.n_clients)[:2]
    expected_ids = ",".join(map(str, sorted(seeded_ids)))
    assert set(comparison.loc[comparison["setting"].ne("clean"), "malicious_client_ids"]) == {expected_ids}
    assert set(comparison["split"]) == {"validation"}

    zero = run_comparison(bundle, replace(config, mal_fraction=0.0), evaluate_test=False)
    clean = zero.loc[zero["setting"].eq("clean"), list(METRICS)].reset_index(drop=True)
    attacked = zero.loc[zero["setting"].eq("attack"), list(METRICS)].reset_index(drop=True)
    pd.testing.assert_frame_equal(clean, attacked)


def test_cli_writes_reproducible_tables_and_plots(synthetic_framingham, tmp_path):
    dataset = tmp_path / "framingham.csv"
    synthetic_framingham.to_csv(dataset, index=False)
    args = [
        "--data", str(dataset), "--seeds", "42", "43", "--epochs", "2",
        "--n-clients", "5", "--mal-fractions", "0", "0.4",
    ]
    first, second = tmp_path / "first", tmp_path / "second"
    for output in (first, second):
        assert main([*args, "--results-dir", str(output)]) == 0
    for name in ("summary.csv", "aggregate.csv", "paired_comparisons.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes()

    summary = pd.read_csv(first / "summary.csv")
    assert len(summary) == 2 * 5 * 3
    assert set(summary.loc[summary["split"].eq("test"), "epoch"]) == {2}
    assert set(pd.read_csv(first / "aggregate.csv")["n_seeds"]) == {2}
    comparisons = pd.read_csv(first / "paired_comparisons.csv")
    assert (comparisons.loc[comparisons["mal_fraction"].eq(0), "accuracy_attack_drop"] == 0).all()
    assert (first / "experiment_config.json").is_file()
    for plot in ("convergence", "final_metrics", "defense_gain"):
        assert (first / "plots" / f"random_{plot}.png").stat().st_size > 0
