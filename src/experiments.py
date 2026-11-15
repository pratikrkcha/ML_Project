"""Run matched comparisons on the local FedDef simulation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.attacks import make_attack_fn
from src.config import Config
from src.data import DataBundle
from src.defense import robust_aggregate
from src.federated import ClientDataset, FederatedResult, run_federated, train_client_update

import matplotlib.pyplot as plt

METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")
PLOT_METRICS = ("accuracy", "f1", "roc_auc")
SETTINGS = {"clean": "Clean FedAvg", "attack": "Attacked FedAvg", "defense": "Attacked + FEDDEF"}
DEFAULT_STRENGTHS = {
    "random": 0.5, "sign_flip": 1.0, "scaling": 3.0, "label_flip": 1.0, "collusion": 0.5,
}
GROUP_COLUMNS = ["setting", "mal_fraction", "attack_type", "attack_strength", "epoch", "split"]


def check_attack_strength(attack_type: str, strength: float) -> None:
    """Bound simulation magnitudes and interpret label-flip strength as a fraction."""
    lower = 1.0 if attack_type == "scaling" else 0.0
    upper = 1.0 if attack_type == "label_flip" else 10.0
    if not lower <= strength <= upper:
        raise ValueError(f"{attack_type} strength must be between {lower:g} and {upper:g}")


def _result_rows(
    result: FederatedResult,
    config: Config,
    setting: str,
    malicious_ids: set[int],
) -> list[dict[str, Any]]:
    is_clean = setting == "clean"
    run = {
        "seed": config.seed,
        "setting": setting,
        "mal_fraction": 0.0 if is_clean else config.mal_fraction,
        "attack_type": "none" if is_clean else config.attack_type,
        "attack_strength": 0.0 if is_clean else config.attack_strength,
        "n_malicious": len(malicious_ids),
        "malicious_client_ids": ",".join(map(str, sorted(malicious_ids))),
    }
    rows = [
        {**run, "epoch": entry["epoch"], "split": "validation",
         **{metric: entry[metric] for metric in METRICS}}
        for entry in result.history
    ]
    if result.final_test_metrics:
        rows.append({
            **run, "epoch": config.epochs, "split": "test",
            **{metric: result.final_test_metrics[metric] for metric in METRICS},
        })
    return rows


def run_comparison(
    bundle: DataBundle,
    config: Config,
    *,
    evaluate_test: bool = True,
    clean_result: FederatedResult | None = None,
) -> pd.DataFrame:
    """Compare clean, attacked, and defended FedAvg on one prepared split.

    Matrix callers may supply a clean result from the same bundle, seed, and
    training configuration to avoid training it again for every scenario.
    """
    check_attack_strength(config.attack_type, config.attack_strength)
    n_malicious = int(config.n_clients * config.mal_fraction)
    client_rng = np.random.default_rng([config.seed, 0])
    malicious_ids = set(client_rng.permutation(config.n_clients)[:n_malicious].tolist())
    attack = make_attack_fn(config.attack_type, strength=config.attack_strength)

    def update_client(
        client: ClientDataset, global_params: Mapping[str, Any], round_idx: int
    ) -> dict[str, Any]:
        if client.client_id not in malicious_ids:
            return train_client_update(client, global_params, round_idx, config=config)

        rng = np.random.default_rng([config.seed, 1, round_idx, client.client_id])
        if config.attack_type == "label_flip":
            # Corrupt only the simulated client's local training labels.
            labels = client.y.copy()
            flipped = rng.choice(len(labels), int(len(labels) * config.attack_strength), replace=False)
            labels.iloc[flipped] = 1 - labels.iloc[flipped]
            return train_client_update(replace(client, y=labels), global_params, round_idx, config=config)

        update = train_client_update(client, global_params, round_idx, config=config)
        return attack(update, rng)

    def aggregate_defended(
        updates: list[dict[str, Any]], weights: list[int], global_params: Mapping[str, Any]
    ) -> dict[str, Any]:
        aggregated, _ = robust_aggregate(
            updates, weights, bundle.X_val, z0=config.z0, theta=config.theta,
            global_params=global_params,
        )
        return aggregated

    if clean_result is None:
        clean_result = run_federated(bundle, config, evaluate_test=evaluate_test, verbose=False)

    attacked = clean_result
    if malicious_ids:
        attacked = run_federated(
            bundle, config, evaluate_test=evaluate_test, verbose=False, client_update=update_client,
        )
    defended = run_federated(
        bundle, config, evaluate_test=evaluate_test, verbose=False,
        client_update=update_client, aggregate_round=aggregate_defended,
    )
    return pd.DataFrame([
        *_result_rows(clean_result, config, "clean", set()),
        *_result_rows(attacked, config, "attack", malicious_ids),
        *_result_rows(defended, config, "defense", malicious_ids),
    ])


def _final_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute paired differences on the final reported split for each seed."""
    split = "test" if metrics["split"].eq("test").any() else "validation"
    final = metrics.loc[metrics["split"].eq(split) & metrics["epoch"].eq(metrics["epoch"].max())]
    clean = final[final["setting"].eq("clean")].set_index("seed")
    keys = ["seed", "mal_fraction", "attack_type", "attack_strength"]
    attacked = final[final["setting"].eq("attack")].set_index(keys)
    defended = final[final["setting"].eq("defense")].set_index(keys)
    pairs = attacked[["n_malicious"]].copy()
    pairs["split"] = split
    for metric in METRICS:
        clean_values = attacked.index.get_level_values("seed").map(clean[metric])
        pairs[f"{metric}_attack_drop"] = clean_values - attacked[metric]
        pairs[f"{metric}_defense_gain"] = defended[metric] - attacked[metric]
    return pairs.reset_index()


def save_experiment_outputs(metrics: pd.DataFrame, results_dir: Path) -> None:
    """Save per-round, aggregated, and paired results with report plots."""
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(results_dir / "summary.csv", index=False)
    aggregations = {
        f"{metric}_{stat}": (metric, stat) for metric in METRICS for stat in ("mean", "std")
    }
    aggregate = metrics.groupby(GROUP_COLUMNS, as_index=False).agg(
        n_seeds=("seed", "nunique"), **aggregations,
    )
    aggregate.to_csv(results_dir / "aggregate.csv", index=False)
    comparisons = _final_comparisons(metrics)
    comparisons.to_csv(results_dir / "paired_comparisons.csv", index=False)

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    clean = aggregate.loc[aggregate["setting"].eq("clean")]
    for attack_type, scenarios in aggregate.loc[aggregate["setting"].ne("clean")].groupby("attack_type"):
        paired = comparisons.loc[comparisons["attack_type"].eq(attack_type)]
        _save_plots(clean, scenarios, paired, plots_dir, attack_type)


def _save_plots(
    clean: pd.DataFrame,
    scenarios: pd.DataFrame,
    paired: pd.DataFrame,
    directory: Path,
    attack_type: str,
) -> None:
    fractions = sorted(scenarios["mal_fraction"].unique())
    shown_fraction = min(fractions, key=lambda fraction: abs(fraction - 0.2))
    validation = scenarios["split"].eq("validation")
    curves = pd.concat([
        clean.loc[clean["split"].eq("validation")],
        scenarios.loc[validation & scenarios["mal_fraction"].eq(shown_fraction)],
    ])
    _plot_settings(
        curves, "epoch", "Communication round", "Validation",
        f"{attack_type} at {shown_fraction:.0%} malicious clients",
        directory / f"{attack_type}_convergence.png",
    )

    split = paired["split"].iloc[0]
    final = scenarios.loc[scenarios["split"].eq(split) & scenarios["epoch"].eq(scenarios["epoch"].max())]
    baseline = clean.loc[clean["split"].eq(split) & clean["epoch"].eq(clean["epoch"].max())]
    final = pd.concat([final, *(baseline.assign(mal_fraction=fraction) for fraction in fractions)])
    _plot_settings(
        final, "mal_fraction", "Malicious-client fraction", split.title(), attack_type,
        directory / f"{attack_type}_final_metrics.png",
    )

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for metric in PLOT_METRICS:
        by_fraction = paired.groupby("mal_fraction")[f"{metric}_defense_gain"].agg(["mean", "std"])
        ax.errorbar(
            by_fraction.index, by_fraction["mean"], yerr=by_fraction["std"].fillna(0),
            marker="o", capsize=3, label=metric.replace("_", " ").upper(),
        )
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.set(xlabel="Malicious-client fraction", ylabel=f"{split.title()} metric gain",
           title=f"FEDDEF minus attacked FedAvg · {attack_type}")
    ax.legend()
    fig.savefig(directory / f"{attack_type}_defense_gain.png", dpi=160)
    plt.close(fig)


def _plot_settings(
    values: pd.DataFrame, x: str, xlabel: str, split: str, title: str, path: Path
) -> None:
    fig, axes = plt.subplots(1, len(PLOT_METRICS), figsize=(11, 3.6), layout="constrained")
    for metric, ax in zip(PLOT_METRICS, axes, strict=True):
        for setting, label in SETTINGS.items():
            row = values.loc[values["setting"].eq(setting)].sort_values(x)
            mean = row[f"{metric}_mean"]
            deviation = row[f"{metric}_std"].fillna(0)
            ax.plot(row[x], mean, marker="o" if x == "mal_fraction" else None, label=label)
            ax.fill_between(row[x], (mean - deviation).clip(0, 1), (mean + deviation).clip(0, 1), alpha=0.12)
        ax.set(xlabel=xlabel, title=metric.replace("_", " ").upper())
    axes[0].set_ylabel(f"{split} score")
    axes[-1].legend(fontsize="small")
    fig.suptitle(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)
