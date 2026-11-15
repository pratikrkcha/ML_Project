"""Federated learning module for FedDef (Task 2).

Implements:
- Seeded client data partitioning with full dataset coverage.
- Weighted FedAvg parameter aggregation.
- Local client training with LogisticRegression (one-shot and true FedAvg).
- Centralized logistic regression reference baseline.
- Server orchestration loop with convergence tracking and metric saving.

Shared parameter representation matches team contract:
    Params = {
        "coef": np.ndarray,      # shape (n_features,), float64
        "intercept": float,      # scalar float
    }
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

# Set MPLCONFIGDIR to avoid warnings in sandboxed environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

from src.config import Config
from src.data import DataBundle, load_data
from src.evaluate import (
    evaluate_model,
    format_metrics,
    save_confusion_matrix_plot,
    save_metrics,
)

FederatedProtocol = Literal["fedavg", "one_shot"]


@dataclass(frozen=True)
class ClientDataset:
    """A client's local dataset shard."""

    client_id: int
    X: pd.DataFrame
    y: pd.Series

    @property
    def n_samples(self) -> int:
        return len(self.X)

    @property
    def positive_count(self) -> int:
        return int((self.y == 1).sum())

    @property
    def negative_count(self) -> int:
        return int((self.y == 0).sum())


@dataclass
class FederatedResult:
    """Complete summary of a federated learning run."""

    protocol: str
    global_params: dict[str, Any]
    history: list[dict[str, Any]]
    final_val_metrics: dict[str, Any]
    final_test_metrics: dict[str, Any]
    config: dict[str, Any]
    n_clients: int


def partition_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    n_clients: int,
    *,
    seed: int = 42,
    stratified: bool = True,
) -> list[ClientDataset]:
    """Partition training data across n_clients with a seeded split.

    Guarantees:
    - Every row is assigned to exactly one client partition (complete coverage, disjoint).
    - When stratified=True, preserves class balance across all clients to avoid
      single-class failure in local logistic regression.
    - Fully deterministic given the seed.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise TypeError("y must be a pandas Series or numpy array")
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} != {len(y)}")
    if n_clients < 1:
        raise ValueError("n_clients must be an integer >= 1")
    if len(X) < n_clients:
        raise ValueError(
            f"Cannot partition {len(X)} samples among {n_clients} clients "
            "(samples must be >= n_clients)"
        )

    rng = np.random.default_rng(seed)
    y_series = pd.Series(y, index=X.index) if not isinstance(y, pd.Series) else y

    if stratified and y_series.nunique() > 1:
        # Partition each class separately and distribute evenly
        client_indices: list[list[Any]] = [[] for _ in range(n_clients)]
        for label in y_series.unique():
            class_indices = y_series.index[y_series == label].to_numpy()
            shuffled_class_indices = rng.permutation(class_indices)
            splits = np.array_split(shuffled_class_indices, n_clients)
            for cid, split_idx in enumerate(splits):
                client_indices[cid].extend(split_idx.tolist())
    else:
        # Simple random shuffle across all samples
        indices = rng.permutation(X.index.to_numpy())
        splits = np.array_split(indices, n_clients)
        client_indices = [split_idx.tolist() for split_idx in splits]

    partitions: list[ClientDataset] = []
    for cid, idxs in enumerate(client_indices):
        # Sort indices to preserve row ordering within each shard
        idx_array = np.array(idxs)
        partitions.append(
            ClientDataset(
                client_id=cid,
                X=X.loc[idx_array].copy(),
                y=y_series.loc[idx_array].copy(),
            )
        )

    return partitions


def fedavg(
    updates: Sequence[Mapping[str, Any]],
    weights: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Weighted FedAvg aggregation across client parameter dictionaries.

    Mathematical formulation:
        w_global = sum_{k=1}^K alpha_k * w_k
        where alpha_k = weight_k / sum_{j=1}^K weight_j

    Parameters
    ----------
    updates : Sequence of parameter dictionaries, each having:
        - "coef": 1D array of shape (n_features,)
        - "intercept": scalar float
    weights : Sequence of non-negative weights (typically client sample counts).
        If None, equal weighting is used.

    Returns
    -------
    Aggregated parameter dictionary with "coef" (1D ndarray) and "intercept" (float).
    """
    if not updates:
        raise ValueError("updates must not be empty")

    n_models = len(updates)
    if weights is None:
        weights_arr = np.ones(n_models, dtype=float)
    else:
        weights_arr = np.asarray(weights, dtype=float)
        if weights_arr.ndim != 1 or len(weights_arr) != n_models:
            raise ValueError(
                f"weights must have shape ({n_models},), got {weights_arr.shape}"
            )
        if np.any(weights_arr < 0.0):
            raise ValueError("All weights must be non-negative")
        if not np.all(np.isfinite(weights_arr)):
            raise ValueError("Weights must not contain NaN or infinite values")

    total_weight = float(np.sum(weights_arr))
    if total_weight <= 0.0:
        raise ValueError("Sum of weights must be strictly positive")

    normalized_weights = weights_arr / total_weight

    # Validate first update structure
    first_coef = np.asarray(updates[0]["coef"], dtype=float)
    if first_coef.ndim != 1:
        raise ValueError(
            f"coef must be a 1D array of shape (n_features,), got {first_coef.shape}"
        )
    n_features = first_coef.shape[0]

    aggregated_coef = np.zeros(n_features, dtype=float)
    aggregated_intercept = 0.0

    for i, (upd, alpha) in enumerate(zip(updates, normalized_weights)):
        if "coef" not in upd or "intercept" not in upd:
            raise KeyError(f"Update {i} must contain both 'coef' and 'intercept'")

        coef_i = np.asarray(upd["coef"], dtype=float)
        if coef_i.shape != (n_features,):
            raise ValueError(
                f"Update {i} coef shape {coef_i.shape} does not match "
                f"expected ({n_features},)"
            )

        intercept_i = float(np.asarray(upd["intercept"]).ravel()[0])

        aggregated_coef += alpha * coef_i
        aggregated_intercept += alpha * intercept_i

    return {
        "coef": aggregated_coef,
        "intercept": float(aggregated_intercept),
    }


def extract_params(model: Any) -> dict[str, Any]:
    """Extract standard Params dictionary from a fitted scikit-learn model."""
    if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
        raise AttributeError("Model must have coef_ and intercept_ attributes")

    coef = np.asarray(model.coef_, dtype=float).reshape(-1).copy()
    intercept = float(np.asarray(model.intercept_).ravel()[0])
    return {
        "coef": coef,
        "intercept": intercept,
    }


def create_model_from_params(
    params: Mapping[str, Any],
    feature_names: Sequence[str] | None = None,
) -> LogisticRegression:
    """Create a scikit-learn LogisticRegression wrapping given parameters."""
    coef = np.asarray(params["coef"], dtype=float).reshape(1, -1)
    intercept = np.array([float(params["intercept"])], dtype=float)

    clf = LogisticRegression()
    clf.classes_ = np.array([0, 1], dtype=int)
    clf.coef_ = coef
    clf.intercept_ = intercept
    clf.n_features_in_ = coef.shape[1]
    if feature_names is not None:
        clf.feature_names_in_ = np.asarray(feature_names, dtype=object)

    return clf


def predict_proba_from_params(
    X: pd.DataFrame | np.ndarray,
    params: Mapping[str, Any],
) -> np.ndarray:
    """Fast vectorised class probability prediction from parameters."""
    X_mat = np.asarray(X, dtype=float)
    coef = np.asarray(params["coef"], dtype=float).reshape(-1)
    intercept = float(params["intercept"])

    logits = X_mat @ coef + intercept
    probs_positive = 1.0 / (1.0 + np.exp(-np.clip(logits, -35.0, 35.0)))
    probs_negative = 1.0 - probs_positive
    return np.column_stack([probs_negative, probs_positive])


def predict_from_params(
    X: pd.DataFrame | np.ndarray,
    params: Mapping[str, Any],
    threshold: float = 0.5,
) -> np.ndarray:
    """Fast vectorised binary label prediction from parameters."""
    probs = predict_proba_from_params(X, params)
    return (probs[:, 1] >= threshold).astype(int)


def train_local_logistic(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    max_iter: int = 1000,
    class_weight: str | dict | None = "balanced",
    random_state: int = 42,
    C: float = 1.0,
) -> dict[str, Any]:
    """Train a standalone local LogisticRegression model on a client shard."""
    y_arr = np.asarray(y, dtype=int).reshape(-1)
    if np.unique(y_arr).size < 2:
        # Fallback for single-class shard
        n_feat = X.shape[1] if hasattr(X, "shape") else len(X[0])
        return {"coef": np.zeros(n_feat), "intercept": 0.0}

    clf = LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
        C=C,
        solver="lbfgs",
    )
    clf.fit(X, y_arr)
    return extract_params(clf)


def train_local_step(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    initial_params: Mapping[str, Any],
    *,
    lr: float = 0.2,
    local_epochs: int = 5,
    class_weight: str | dict | None = "balanced",
    C: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    """True FedAvg local step: optimize from initial global parameters.

    Clients start from the broadcasted global parameters and perform
    local gradient descent steps matching the balanced log-loss objective.
    """
    X_mat = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    n_samples, n_features = X_mat.shape

    # Handle degenerate single-class shard safely
    if np.unique(y_arr).size < 2:
        return {
            "coef": np.asarray(initial_params["coef"], dtype=float).copy(),
            "intercept": float(initial_params["intercept"]),
        }

    # Compute sample weights for balanced classification
    if class_weight == "balanced":
        sample_w = compute_sample_weight("balanced", y_arr)
    else:
        sample_w = np.ones(n_samples, dtype=float)

    sum_w = float(np.sum(sample_w))
    sample_w_norm = sample_w / sum_w

    # Regularization penalty matching scikit-learn: (1 / (C * sum_w)) * w
    reg_lambda = 1.0 / (C * sum_w) if (C > 0 and sum_w > 0) else 0.0

    coef = np.asarray(initial_params["coef"], dtype=float).copy()
    intercept = float(initial_params["intercept"])

    # Local optimization steps
    for _ in range(local_epochs):
        logits = X_mat @ coef + intercept
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -35.0, 35.0)))
        errors = sample_w_norm * (probs - y_arr)

        grad_coef = X_mat.T @ errors + reg_lambda * coef
        grad_intercept = float(np.sum(errors))

        coef -= lr * grad_coef
        intercept -= lr * grad_intercept

    return {
        "coef": coef,
        "intercept": float(intercept),
    }


def train_centralized_model(
    bundle: DataBundle,
    *,
    max_iter: int = 1000,
    class_weight: str | dict | None = "balanced",
    random_state: int = 42,
    C: float = 1.0,
) -> tuple[LogisticRegression, dict[str, Any], dict[str, Any]]:
    """Train centralized logistic regression on full training set as a reference.

    Returns (model, val_metrics, test_metrics).
    """
    model = LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
        C=C,
        solver="lbfgs",
    )
    model.fit(bundle.X_train, bundle.y_train)

    val_metrics = evaluate_model(model, bundle.X_val, bundle.y_val)
    test_metrics = evaluate_model(model, bundle.X_test, bundle.y_test)
    return model, val_metrics, test_metrics


def train_client_update(
    client: ClientDataset,
    global_params: Mapping[str, Any],
    round_idx: int,
    *,
    config: Config,
    local_epochs: int = 5,
) -> dict[str, Any]:
    """Train a client's update using the baseline FedAvg hyperparameters."""
    return train_local_step(
        client.X,
        client.y,
        global_params,
        lr=config.lr,
        local_epochs=local_epochs,
        class_weight="balanced",
        seed=config.seed + round_idx * 100 + client.client_id,
    )


def run_federated(
    bundle: DataBundle,
    config: Config | None = None,
    *,
    protocol: FederatedProtocol = "fedavg",
    local_epochs: int = 5,
    evaluate_test: bool = True,
    verbose: bool = True,
    client_update: Callable[[ClientDataset, Mapping[str, Any], int], dict[str, Any]] | None = None,
    aggregate_round: Callable[
        [list[dict[str, Any]], list[int], Mapping[str, Any]], dict[str, Any]
    ] | None = None,
) -> FederatedResult:
    """Execute the federated learning baseline or a custom FedAvg experiment.

    Protocols:
    - 'fedavg' (preferred, default): True FedAvg. Clients initialize each round
      from current global parameters and take local_epochs gradient steps with lr.
    - 'one_shot': Each client trains a full LogisticRegression(max_iter=1000)
      to convergence, then server performs one-shot weighted FedAvg.

    Parameters
    ----------
    bundle : Prepared DataBundle containing X_train, y_train, X_val, y_val, X_test, y_test.
    config : Experiment configuration. If None, default Config is used.
    protocol : 'fedavg' or 'one_shot'.
    local_epochs : Number of local steps per round for true FedAvg.
    evaluate_test : Whether to evaluate the final global model on the test split.
    verbose : Whether to print round-by-round progress.
    client_update : Optional FedAvg client trainer, receiving the broadcast model and round.
    aggregate_round : Optional FedAvg aggregator, receiving updates, counts and the prior model.

    Returns
    -------
    FederatedResult containing history, final metrics, and parameters.
    """
    if config is None:
        config = Config()
    if protocol == "one_shot" and (client_update is not None or aggregate_round is not None):
        raise ValueError("Custom training and aggregation require the fedavg protocol")

    n_clients = config.n_clients
    seed = config.seed
    epochs = config.epochs

    # Partition training data
    partitions = partition_dataset(
        bundle.X_train,
        bundle.y_train,
        n_clients=n_clients,
        seed=seed,
        stratified=True,
    )
    client_weights = [part.n_samples for part in partitions]

    n_features = bundle.n_features
    feature_names = bundle.feature_names

    # Global parameter initialization
    global_params = {
        "coef": np.zeros(n_features, dtype=float),
        "intercept": 0.0,
    }

    history: list[dict[str, Any]] = []

    if verbose:
        print("=" * 68)
        print(f"FedDef Baseline Server — Protocol: {protocol.upper()}")
        print(f"Clients: {n_clients} | Rounds: {epochs if protocol == 'fedavg' else 1} | lr: {config.lr} | Seed: {seed}")
        print("=" * 68)

    if protocol == "one_shot":
        # One-shot FedAvg: each client fits to convergence with max_iter=1000
        client_updates = []
        for client in partitions:
            upd = train_local_logistic(
                client.X,
                client.y,
                max_iter=1000,
                class_weight="balanced",
                random_state=seed + client.client_id,
            )
            client_updates.append(upd)

        global_params = fedavg(client_updates, client_weights)
        eval_model = create_model_from_params(global_params, feature_names)
        val_m = evaluate_model(eval_model, bundle.X_val, bundle.y_val)

        history.append({
            "epoch": 1,
            "round": 1,
            "accuracy": val_m["accuracy"],
            "precision": val_m["precision"],
            "recall": val_m["recall"],
            "f1": val_m["f1"],
            "roc_auc": val_m["roc_auc"],
        })
        if verbose:
            print(f"One-shot round complete | Val Acc: {val_m['accuracy']:.4f} | F1: {val_m['f1']:.4f} | ROC-AUC: {val_m['roc_auc']:.4f}")

    elif protocol == "fedavg":
        # True FedAvg: multi-round loop
        update_client = client_update or partial(
            train_client_update, config=config, local_epochs=local_epochs
        )

        for round_idx in range(1, epochs + 1):
            client_updates = [
                update_client(client, global_params, round_idx) for client in partitions
            ]

            # Server aggregates client updates
            if aggregate_round is None:
                global_params = fedavg(client_updates, client_weights)
            else:
                global_params = aggregate_round(client_updates, client_weights, global_params)

            # Evaluate global model on validation set
            eval_model = create_model_from_params(global_params, feature_names)
            val_m = evaluate_model(eval_model, bundle.X_val, bundle.y_val)

            record = {
                "epoch": round_idx,
                "round": round_idx,
                "accuracy": val_m["accuracy"],
                "precision": val_m["precision"],
                "recall": val_m["recall"],
                "f1": val_m["f1"],
                "roc_auc": val_m["roc_auc"],
            }
            history.append(record)

            if verbose and (round_idx == 1 or round_idx % 10 == 0 or round_idx == epochs):
                auc_str = f"{val_m['roc_auc']:.4f}" if val_m["roc_auc"] is not None else "n/a"
                print(
                    f"Round {round_idx:3d}/{epochs:3d} | "
                    f"Val Acc: {val_m['accuracy']:6.2%} | "
                    f"F1: {val_m['f1']:.4f} | "
                    f"ROC-AUC: {auc_str}"
                )
    else:
        raise ValueError(f"Unknown protocol: {protocol!r}")

    final_eval_model = create_model_from_params(global_params, feature_names)
    final_val_metrics = evaluate_model(final_eval_model, bundle.X_val, bundle.y_val)

    final_test_metrics: dict[str, Any] = {}
    if evaluate_test:
        final_test_metrics = evaluate_model(final_eval_model, bundle.X_test, bundle.y_test)

    return FederatedResult(
        protocol=protocol,
        global_params=global_params,
        history=history,
        final_val_metrics=final_val_metrics,
        final_test_metrics=final_test_metrics,
        config=config.to_dict(),
        n_clients=n_clients,
    )


def save_convergence_plot(
    history: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    centralized_val_metrics: Mapping[str, Any] | None = None,
    title: str = "FedAvg Baseline Convergence",
) -> Path:
    """Save convergence curves (accuracy, F1, ROC-AUC) over rounds as a PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rounds = [entry["epoch"] for entry in history]
    accuracies = [entry["accuracy"] for entry in history]
    f1s = [entry["f1"] for entry in history]
    aucs = [entry.get("roc_auc", np.nan) for entry in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Accuracy plot
    axes[0].plot(rounds, accuracies, marker="o", markersize=3, label="FedAvg (Val)", color="#1f77b4")
    if centralized_val_metrics and "accuracy" in centralized_val_metrics:
        cent_acc = centralized_val_metrics["accuracy"]
        axes[0].axhline(cent_acc, color="#d62728", linestyle="--", label=f"Centralized ({cent_acc:.3f})")
    axes[0].set_title("Accuracy vs Round")
    axes[0].set_xlabel("Communication Round")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right")

    # F1 score plot
    axes[1].plot(rounds, f1s, marker="s", markersize=3, label="FedAvg (Val)", color="#2ca02c")
    if centralized_val_metrics and "f1" in centralized_val_metrics:
        cent_f1 = centralized_val_metrics["f1"]
        axes[1].axhline(cent_f1, color="#d62728", linestyle="--", label=f"Centralized ({cent_f1:.3f})")
    axes[1].set_title("F1 Score vs Round")
    axes[1].set_xlabel("Communication Round")
    axes[1].set_ylabel("F1 Score (Positive Class)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower right")

    # ROC-AUC plot
    axes[2].plot(rounds, aucs, marker="^", markersize=3, label="FedAvg (Val)", color="#ff7f0e")
    if centralized_val_metrics and "roc_auc" in centralized_val_metrics:
        cent_auc = centralized_val_metrics["roc_auc"]
        if cent_auc is not None:
            axes[2].axhline(cent_auc, color="#d62728", linestyle="--", label=f"Centralized ({cent_auc:.3f})")
    axes[2].set_title("ROC-AUC vs Round")
    axes[2].set_xlabel("Communication Round")
    axes[2].set_ylabel("ROC-AUC")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="lower right")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_baseline_artifacts(
    fed_result: FederatedResult,
    centralized_val: dict[str, Any],
    centralized_test: dict[str, Any],
    results_dir: str | Path = "results/baseline",
) -> dict[str, Path]:
    """Persist all baseline results, metrics, plots, and convergence CSV."""
    dir_path = Path(results_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, Path] = {}

    # Save FedAvg metrics
    saved_paths["fedavg_val_metrics"] = save_metrics(
        fed_result.final_val_metrics,
        dir_path / "fedavg_val_metrics.json",
    )
    if fed_result.final_test_metrics:
        saved_paths["fedavg_test_metrics"] = save_metrics(
            fed_result.final_test_metrics,
            dir_path / "fedavg_test_metrics.json",
        )
        saved_paths["fedavg_test_cm"] = save_confusion_matrix_plot(
            fed_result.final_test_metrics,
            dir_path / "fedavg_test_confusion_matrix.png",
            title="FedAvg Baseline — Held-Out Test Set",
        )

    # Save Centralized metrics
    saved_paths["centralized_val_metrics"] = save_metrics(
        centralized_val,
        dir_path / "centralized_val_metrics.json",
    )
    saved_paths["centralized_test_metrics"] = save_metrics(
        centralized_test,
        dir_path / "centralized_test_metrics.json",
    )
    saved_paths["centralized_test_cm"] = save_confusion_matrix_plot(
        centralized_test,
        dir_path / "centralized_test_confusion_matrix.png",
        title="Centralized Reference — Held-Out Test Set",
    )

    # Save convergence history to CSV
    df_history = pd.DataFrame(fed_result.history)
    conv_csv_path = dir_path / "convergence.csv"
    df_history.to_csv(conv_csv_path, index=False)
    saved_paths["convergence_csv"] = conv_csv_path

    # Save convergence plot
    conv_png_path = dir_path / "convergence.png"
    save_convergence_plot(
        fed_result.history,
        conv_png_path,
        centralized_val_metrics=centralized_val,
        title=f"FedAvg ({fed_result.protocol.upper()}) Convergence vs Centralized Reference",
    )
    saved_paths["convergence_png"] = conv_png_path

    # Compute comparison summary and verify tolerance
    fed_test_acc = fed_result.final_test_metrics.get("accuracy", 0.0)
    cent_test_acc = centralized_test.get("accuracy", 0.0)
    acc_diff = fed_test_acc - cent_test_acc

    fed_test_f1 = fed_result.final_test_metrics.get("f1", 0.0)
    cent_test_f1 = centralized_test.get("f1", 0.0)
    f1_diff = fed_test_f1 - cent_test_f1

    fed_test_auc = fed_result.final_test_metrics.get("roc_auc")
    cent_test_auc = centralized_test.get("roc_auc")
    auc_diff = (fed_test_auc - cent_test_auc) if (fed_test_auc is not None and cent_test_auc is not None) else None

    # Suggested tolerance from acceptance criteria: within +/- 2-3% accuracy
    TOLERANCE_ACC = 0.03
    within_tolerance = abs(acc_diff) <= TOLERANCE_ACC

    comparison = {
        "protocol": fed_result.protocol,
        "n_clients": fed_result.n_clients,
        "seed": fed_result.config.get("seed", 42),
        "tolerance_threshold": TOLERANCE_ACC,
        "tolerance_satisfied": bool(within_tolerance),
        "test_comparison": {
            "centralized_accuracy": float(cent_test_acc),
            "fedavg_accuracy": float(fed_test_acc),
            "accuracy_difference": float(acc_diff),
            "centralized_f1": float(cent_test_f1),
            "fedavg_f1": float(fed_test_f1),
            "f1_difference": float(f1_diff),
            "centralized_roc_auc": float(cent_test_auc) if cent_test_auc is not None else None,
            "fedavg_roc_auc": float(fed_test_auc) if fed_test_auc is not None else None,
            "roc_auc_difference": float(auc_diff) if auc_diff is not None else None,
        },
        "config": fed_result.config,
    }

    comp_path = dir_path / "comparison.json"
    with comp_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)
        handle.write("\n")
    saved_paths["comparison_json"] = comp_path

    # Save run config
    cfg_path = dir_path / "baseline_config.json"
    with cfg_path.open("w", encoding="utf-8") as handle:
        json.dump(fed_result.config, handle, indent=2)
        handle.write("\n")
    saved_paths["baseline_config"] = cfg_path

    return saved_paths


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for running the federated baseline directly."""
    parser = argparse.ArgumentParser(
        description="FedDef Task 2: Clean FedAvg baseline & Centralized Reference",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to framingham.csv dataset.",
    )
    parser.add_argument(
        "--protocol",
        choices=["fedavg", "one_shot"],
        default="fedavg",
        help="FL protocol to run: true 'fedavg' (default) or 'one_shot'.",
    )
    parser.add_argument(
        "--n-clients",
        type=int,
        default=20,
        help="Number of federated clients (default: 20).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of communication rounds (default: 50).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.2,
        help="Client learning rate (default: 0.2).",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=5,
        help="Number of local epochs per round (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/baseline"),
        help="Output directory for baseline artifacts (default: results/baseline).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for running the federated baseline."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Resolve dataset path
    data_path = args.data
    if data_path is None:
        for candidate in (Path("data/framingham.csv"), Path("framingham.csv")):
            if candidate.exists():
                data_path = candidate
                break
        if data_path is None:
            data_path = Path("data/framingham.csv")

    if not data_path.exists():
        print(f"Dataset not found at {data_path}. Please place framingham.csv in data/.", flush=True)
        return 1

    config = Config(
        seed=args.seed,
        n_clients=args.n_clients,
        epochs=args.epochs,
        lr=args.lr,
        data_path=data_path,
        results_dir=args.results_dir.parent,
    )

    print("\n[1/3] Loading and preparing Framingham dataset...")
    bundle = load_data(
        config.data_path,
        target_col=config.target_col,
        test_size=config.test_size,
        val_size=config.val_size,
        random_state=config.seed,
    )
    print(f"Loaded: {bundle.n_train} train rows, {bundle.n_val} val rows, {bundle.n_test} test rows, {bundle.n_features} features.")

    print("\n[2/3] Training centralized reference model...")
    cent_model, cent_val, cent_test = train_centralized_model(
        bundle,
        max_iter=1000,
        class_weight="balanced",
        random_state=config.seed,
    )
    print("Centralized Test Set Metrics:")
    print(format_metrics(cent_test))

    print(f"\n[3/3] Running {args.protocol.upper()} server loop...")
    fed_result = run_federated(
        bundle,
        config=config,
        protocol=args.protocol,
        local_epochs=args.local_epochs,
        evaluate_test=True,
        verbose=True,
    )
    print("\nFedAvg Final Test Set Metrics:")
    print(format_metrics(fed_result.final_test_metrics))

    # Save artifacts
    saved = save_baseline_artifacts(
        fed_result,
        cent_val,
        cent_test,
        results_dir=args.results_dir,
    )

    # Print comparison
    cent_acc = cent_test["accuracy"]
    fed_acc = fed_result.final_test_metrics["accuracy"]
    cent_f1 = cent_test["f1"]
    fed_f1 = fed_result.final_test_metrics["f1"]
    cent_auc = cent_test.get("roc_auc")
    fed_auc = fed_result.final_test_metrics.get("roc_auc")

    print("\n" + "=" * 68)
    print("BASELINE COMPARISON: CENTRALIZED vs FEDAVG (TEST SET)")
    print("=" * 68)
    print(f"Metric        | Centralized | FedAvg      | Difference")
    print(f"------------- | ----------- | ----------- | ----------")
    print(f"Accuracy      | {cent_acc:11.4f} | {fed_acc:11.4f} | {fed_acc - cent_acc:+10.4f}")
    print(f"F1 Score      | {cent_f1:11.4f} | {fed_f1:11.4f} | {fed_f1 - cent_f1:+10.4f}")
    auc_cent_str = f"{cent_auc:11.4f}" if cent_auc is not None else "        n/a"
    auc_fed_str = f"{fed_auc:11.4f}" if fed_auc is not None else "        n/a"
    auc_diff_str = f"{fed_auc - cent_auc:+10.4f}" if (fed_auc is not None and cent_auc is not None) else "       n/a"
    print(f"ROC-AUC       | {auc_cent_str} | {auc_fed_str} | {auc_diff_str}")
    print("-" * 68)

    tolerance = 0.03
    within_tolerance = abs(fed_acc - cent_acc) <= tolerance
    status_str = "PASSED (within +/- 3.0% tolerance)" if within_tolerance else "FAILED (exceeds tolerance)"
    print(f"Acceptance Criteria Tolerance Check: {status_str}")
    print(f"Artifacts saved under: {args.results_dir.resolve()}")
    for key, path in saved.items():
        print(f"  - {key}: {path.name}")
    print("=" * 68 + "\n")

    return 0 if within_tolerance else 1


if __name__ == "__main__":
    raise SystemExit(main())
