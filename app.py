"""FedDef Streamlit app for the baseline, attack, and defense comparisons.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

from src.config import Config
from src.data import DataBundle, load_data, prepare_data
from src.evaluate import evaluate_model, format_metrics
from src.experiments import DEFAULT_STRENGTHS, SETTINGS, run_comparison
from src.federated import (
    run_federated,
    save_baseline_artifacts,
    train_centralized_model,
)

st.set_page_config(
    page_title="FedDef",
    page_icon="🛡️",
    layout="wide",
)

DEFAULT_DATA_PATH = Path("data/framingham.csv")
TARGET_COLUMN = "TenYearCHD"


def _load_uploaded(
    uploaded_file,
    *,
    seed: int,
    test_size: float,
    val_size: float,
) -> DataBundle:
    frame = pd.read_csv(uploaded_file)
    return prepare_data(
        frame,
        target_col=TARGET_COLUMN,
        test_size=test_size,
        val_size=val_size,
        random_state=seed,
        source=getattr(uploaded_file, "name", "uploaded CSV"),
    )


def _load_default(
    *,
    seed: int,
    test_size: float,
    val_size: float,
) -> DataBundle | None:
    if not DEFAULT_DATA_PATH.exists():
        return None
    return load_data(
        DEFAULT_DATA_PATH,
        target_col=TARGET_COLUMN,
        test_size=test_size,
        val_size=val_size,
        random_state=seed,
    )


st.title("FedDef — Federated Learning Robustness Demo")
st.caption(
    "Compare clean FedAvg, simulated model poisoning, and anomaly-aware aggregation."
)

with st.sidebar:
    st.header("Data")
    uploaded_file = st.file_uploader("Upload Framingham CSV", type=["csv"])
    seed = int(
        st.number_input(
            "Random seed",
            min_value=0,
            max_value=100_000,
            value=42,
            step=1,
        )
    )
    test_size = float(
        st.slider(
            "Test fraction",
            min_value=0.10,
            max_value=0.40,
            value=0.20,
            step=0.05,
        )
    )
    val_size = float(
        st.slider(
            "Validation fraction",
            min_value=0.05,
            max_value=0.30,
            value=0.10,
            step=0.05,
        )
    )
    use_default = st.checkbox(
        "Use data/framingham.csv if no upload",
        value=True,
    )
    st.divider()
    st.info(
        "The test split is reserved for final evaluation. The defense will use "
        "the validation split only."
    )

if test_size + val_size >= 0.9:
    st.error("Test fraction + validation fraction must be less than 0.9.")
    st.stop()

bundle: DataBundle | None = None

if uploaded_file is not None:
    try:
        bundle = _load_uploaded(
            uploaded_file,
            seed=seed,
            test_size=test_size,
            val_size=val_size,
        )
    except (ValueError, TypeError, pd.errors.ParserError) as exc:
        st.error(f"Could not prepare the uploaded CSV: {exc}")
elif use_default:
    try:
        bundle = _load_default(
            seed=seed,
            test_size=test_size,
            val_size=val_size,
        )
    except (ValueError, TypeError) as exc:
        st.error(f"Could not prepare data/framingham.csv: {exc}")

if bundle is None:
    st.info(
        "Upload a Framingham CSV in the sidebar, or place it at "
        "`data/framingham.csv`. See `data/README.md` for the expected columns."
    )
    st.stop()

metadata = bundle.metadata
st.success(
    f"Loaded **{metadata.get('source') or 'dataset'}** — "
    f"{metadata['n_rows_after_target_drop']:,} rows, "
    f"{bundle.n_features} features."
)

metric_columns = st.columns(3)
for column, split_name in zip(metric_columns, ("train", "val", "test")):
    column.metric(
        f"{split_name.capitalize()} rows",
        f"{metadata['split_sizes'][split_name]:,}",
    )
    column.caption(f"{metadata['positive_rate'][split_name]:.1%} positive")

data_tab, baseline_tab, attack_tab, defense_tab = st.tabs(
    [
        "Data",
        "Baseline (Task 2)",
        "Attack (Task 4)",
        "Defense (Task 3)",
    ]
)

with data_tab:
    st.subheader("Prepared training features")
    st.caption(
        "Features are median/mode imputed, one-hot encoded, and standardised. "
        "Preprocessing is fitted on the training split only."
    )
    st.dataframe(bundle.X_train.head(10), width="stretch")

    with st.expander("Dataset metadata"):
        st.json(metadata)

    st.subheader("Centralized smoke check")
    st.caption(
        "This is a data/evaluation sanity check, not the Task 2 FedAvg "
        "baseline. It trains one logistic regression on the training split and "
        "reports validation metrics."
    )
    if st.button("Train smoke-check model"):
        with st.spinner("Training logistic regression..."):
            model = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=seed,
            )
            model.fit(bundle.X_train, bundle.y_train)
            metrics = evaluate_model(model, bundle.X_val, bundle.y_val)

        st.text(format_metrics(metrics))
        st.dataframe(
            pd.DataFrame(
                metrics["confusion_matrix"],
                index=["True 0", "True 1"],
                columns=["Predicted 0", "Predicted 1"],
            ),
            width="stretch",
        )
        st.caption(
            "Validation split only. The test split is not touched by this "
            "smoke check."
        )

with baseline_tab:
    st.subheader("Clean FedAvg Baseline (Task 2 — Pratik Rakhecha)")
    st.caption(
        "Demonstrates a clean, no-attack FedAvg baseline and proves its "
        "convergence matches a centralized logistic regression within tolerance (±2–3%)."
    )

    baseline_dir = Path("results/baseline")
    comp_file = baseline_dir / "comparison.json"
    conv_png = baseline_dir / "convergence.png"

    # Sidebar / Tab controls for baseline
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)
    with ctrl_col1:
        fl_protocol = st.selectbox(
            "FL Protocol",
            options=["fedavg", "one_shot"],
            index=0,
            help="Preferred: true FedAvg. One-shot averages independently trained client models.",
        )
    with ctrl_col2:
        fl_clients = st.number_input("Number of clients", min_value=2, max_value=50, value=20, step=1)
    with ctrl_col3:
        fl_epochs = st.number_input("Rounds / Epochs", min_value=1, max_value=100, value=50, step=5)
    with ctrl_col4:
        fl_lr = st.number_input("Learning rate", min_value=0.01, max_value=1.0, value=0.2, step=0.05)

    if st.button("Run FedAvg Baseline Live", type="primary"):
        with st.spinner("Training centralized reference and running federated baseline..."):
            fed_cfg = Config(
                seed=seed,
                test_size=test_size,
                val_size=val_size,
                target_col=TARGET_COLUMN,
                n_clients=int(fl_clients),
                epochs=int(fl_epochs),
                lr=float(fl_lr),
            )
            cent_model, cent_val, cent_test = train_centralized_model(
                bundle,
                max_iter=1000,
                class_weight="balanced",
                random_state=seed,
            )
            fed_res = run_federated(
                bundle,
                config=fed_cfg,
                protocol=fl_protocol,
                evaluate_test=True,
                verbose=False,
            )
            save_baseline_artifacts(
                fed_res,
                cent_val,
                cent_test,
                results_dir=baseline_dir,
            )
            st.session_state["baseline_run_done"] = True
            st.success("Baseline training completed and saved!")

    # Display comparison if available
    if comp_file.exists():
        with comp_file.open("r", encoding="utf-8") as f:
            comp_data = json.load(f)

        test_cmp = comp_data.get("test_comparison", {})
        cent_acc = test_cmp.get("centralized_accuracy", 0.0)
        fed_acc = test_cmp.get("fedavg_accuracy", 0.0)
        acc_diff = test_cmp.get("accuracy_difference", 0.0)

        cent_f1 = test_cmp.get("centralized_f1", 0.0)
        fed_f1 = test_cmp.get("fedavg_f1", 0.0)
        f1_diff = test_cmp.get("f1_difference", 0.0)

        cent_auc = test_cmp.get("centralized_roc_auc", 0.0)
        fed_auc = test_cmp.get("fedavg_roc_auc", 0.0)
        auc_diff = test_cmp.get("roc_auc_difference", 0.0)

        st.subheader("Held-Out Test Set: Centralized vs FedAvg")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Accuracy (FedAvg)", f"{fed_acc:6.2%}", delta=f"{acc_diff:+5.2%}")
        m_col2.metric("F1 Score (FedAvg)", f"{fed_f1:6.2%}", delta=f"{f1_diff:+5.2%}")
        if cent_auc is not None and fed_auc is not None:
            m_col3.metric("ROC-AUC (FedAvg)", f"{fed_auc:.4f}", delta=f"{auc_diff:+.4f}")

        is_passed = comp_data.get("tolerance_satisfied", False)
        if is_passed:
            st.success("✅ Acceptance Criteria Satisfied: FedAvg accuracy is within ±3.0% of centralized reference.")
        else:
            st.warning("⚠️ FedAvg accuracy difference exceeds ±3.0% tolerance.")

        # Comparison table
        cmp_df = pd.DataFrame(
            [
                {"Metric": "Accuracy", "Centralized": f"{cent_acc:.4f}", "FedAvg": f"{fed_acc:.4f}", "Difference": f"{acc_diff:+.4f}"},
                {"Metric": "F1 Score", "Centralized": f"{cent_f1:.4f}", "FedAvg": f"{fed_f1:.4f}", "Difference": f"{f1_diff:+.4f}"},
                {"Metric": "ROC-AUC", "Centralized": f"{cent_auc:.4f}" if cent_auc else "n/a", "FedAvg": f"{fed_auc:.4f}" if fed_auc else "n/a", "Difference": f"{auc_diff:+.4f}" if auc_diff else "n/a"},
            ]
        )
        st.dataframe(cmp_df, hide_index=True, width="stretch")

        if conv_png.exists():
            st.subheader("Convergence Curve (Accuracy, F1, ROC-AUC over Rounds)")
            st.image(str(conv_png), caption="Validation metrics across communication rounds compared to centralized reference.", use_container_width=True)

        # Show confusion matrices
        cm_fed_png = baseline_dir / "fedavg_test_confusion_matrix.png"
        cm_cent_png = baseline_dir / "centralized_test_confusion_matrix.png"
        if cm_fed_png.exists() and cm_cent_png.exists():
            st.subheader("Confusion Matrices (Held-Out Test Set)")
            cm_col1, cm_col2 = st.columns(2)
            with cm_col1:
                st.image(str(cm_fed_png), caption="FedAvg Confusion Matrix", use_container_width=True)
            with cm_col2:
                st.image(str(cm_cent_png), caption="Centralized Reference Confusion Matrix", use_container_width=True)
    else:
        st.info("No baseline results found yet. Click 'Run FedAvg Baseline Live' above or run `python main.py --baseline`.")

with attack_tab:
    st.subheader("Compare clean, attacked, and defended FedAvg")
    attack_type = st.selectbox(
        "Simulated attack", options=list(DEFAULT_STRENGTHS),
        format_func=lambda name: name.replace("_", " ").capitalize(),
    )
    with st.form("attack_comparison"):
        first, second = st.columns(2)
        with first:
            n_clients = st.number_input("Clients", 2, 100, 20)
            mal_fraction = st.slider("Malicious-client fraction", 0.0, 0.9, 0.2, 0.05)
            strength = st.number_input(
                "Attack strength", min_value=1.0 if attack_type == "scaling" else 0.0,
                max_value=1.0 if attack_type == "label_flip" else 10.0,
                value=DEFAULT_STRENGTHS[attack_type], step=0.05,
                key=f"strength_{attack_type}",
            )
        with second:
            epochs = st.number_input("Communication rounds", 1, 100, 20)
            lr = st.number_input("Client learning rate", 0.01, 1.0, 0.20, 0.01)
        submitted = st.form_submit_button("Run comparison")

    if submitted:
        try:
            config = Config(
                seed=seed, test_size=test_size, val_size=val_size,
                n_clients=int(n_clients), mal_fraction=mal_fraction, epochs=int(epochs),
                lr=lr, attack_type=attack_type, attack_strength=strength,
            )
            with st.spinner("Comparing the three settings on the validation split…"):
                comparison = run_comparison(bundle, config, evaluate_test=False)
            st.session_state["attack_comparison_result"] = (comparison, config)
        except ValueError as exc:
            st.error(str(exc))

    if "attack_comparison_result" in st.session_state:
        comparison, run_config = st.session_state["attack_comparison_result"]
        actual_malicious = int(run_config.n_clients * run_config.mal_fraction)
        st.caption(
            f"Last run: seed {run_config.seed}, {run_config.attack_type.replace('_', ' ')}, "
            f"{actual_malicious}/{run_config.n_clients} malicious clients. Validation metrics."
        )
        metric = st.selectbox("Comparison metric", ["accuracy", "f1", "roc_auc"])
        lines = comparison.pivot(index="epoch", columns="setting", values=metric)
        st.line_chart(lines.rename(columns=SETTINGS))
        st.download_button(
            "Download comparison CSV", comparison.to_csv(index=False),
            file_name="feddef_comparison.csv", mime="text/csv",
        )

with defense_tab:
    st.subheader("FEDDEF: Anomaly-Aware Robust Aggregation (Task 3)")
    st.caption("Implemented by Kunal V Kurundwadkar (IIT2024177) — based on maskeensingh/FEDDEF")

    defense_dir = Path("results/defense")
    conv_png = defense_dir / "defense_convergence.png"
    excl_png = defense_dir / "defense_exclusions.png"
    val_json = defense_dir / "defense_val_metrics.json"
    test_json = defense_dir / "defense_test_metrics.json"
    cm_png = defense_dir / "defense_test_confusion_matrix.png"

    if val_json.exists():
        import json
        with val_json.open() as fh:
            d_val = json.load(fh)
        d_test = {}
        if test_json.exists():
            with test_json.open() as fh:
                d_test = json.load(fh)

        st.success("Artifacts loaded from `results/defense/`")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Validation Accuracy", f"{d_val.get('accuracy', 0.0):.4f}")
        col2.metric("Validation F1", f"{d_val.get('f1', 0.0):.4f}")
        if d_test:
            col3.metric("Test Accuracy", f"{d_test.get('accuracy', 0.0):.4f}")
            col4.metric("Test ROC-AUC", f"{d_test.get('roc_auc', 0.0):.4f}")

        if conv_png.exists():
            st.image(str(conv_png), caption="Defense Convergence over Communication Rounds", use_container_width=True)
        if excl_png.exists():
            st.image(str(excl_png), caption="Clients Excluded per Round by Anomaly Detection", use_container_width=True)
        if cm_png.exists():
            st.image(str(cm_png), caption="Defense Confusion Matrix (Test Set)", use_container_width=True)
    else:
        st.info("Run `python -m src.defense` or `python main.py --defense` to generate and view defense artifacts.")
        st.markdown(
            """
            **Algorithm overview:**
            1. Per communication round, each client submits local parameters.
            2. Server computes prediction probabilities on the server validation split.
            3. Prediction vectors are standard-scored (z-scores) across clients per sample.
            4. Anomaly score accumulates whenever $|z| \\ge z_0$ (default 1.0).
            5. Clients with normalized score $\\ge \\theta$ (default 0.25) are excluded.
            6. Surviving clients are aggregated via sample-weighted FedAvg.
            """
        )
