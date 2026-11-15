# FedDef

**Robust Aggregation Against Model Poisoning Attacks in Federated Learning**

FedDef simulates multiple clients jointly training a logistic-regression model
for ten-year coronary heart disease risk prediction, without sharing raw
patient records. A configurable fraction of clients behave maliciously and
submit corrupted model updates. The project compares:

1. clean FedAvg (no attackers)
2. FedAvg under attack, with no defense
3. FedAvg under attack, with an anomaly-aware defense

The defense scores each client's contribution using a robust z-score over
validation-set predictions, accumulates a suspicion score, and down-weights or
excludes suspicious clients before aggregation.

## Current status

- **Task 1 (foundation) is implemented** by Kunal Khandelwal.
- **Task 2 (clean FedAvg baseline + centralized reference) is implemented** by Pratik Rakhecha.

Task 2 deliverables completed:
- `src/federated.py`: seeded client partitioning, weighted `fedavg()`, true FedAvg server loop, one-shot comparison, and centralized reference.
- Acceptance criteria satisfied: FedAvg accuracy (66.86%) matches centralized reference (66.98%) within -0.12% on the held-out test set (well within the ±2–3% tolerance).
- Full artifact output under `results/baseline/`: metrics JSON, convergence CSV/PNG, and confusion matrices.
- 10 comprehensive tests in `tests/test_federated.py` (all 70 tests in the repository pass).
- App integration in `app.py` for live/persisted baseline comparison.

- **Task 3 (z-score defense and validation protocol)** is implemented by Kunal V Kurundwadkar.
- **Task 4 (experiment runner, comparisons, and plots)** is implemented by Dewansh Khandelwal after the Task 2/4 swap.

See [`team_plan.md`](team_plan.md) for the workstreams and owners.

## Repository layout

```text
main.py                 CLI: foundation, baseline, and defense runs
run_experiments.py      CLI: matched core or extended experiment matrix
app.py                  Streamlit baseline, attack, and defense app
src/
  config.py             shared, validated experiment configuration
  data.py               loading, cleaning, splitting, scaling
  evaluate.py           metrics, JSON saving, confusion-matrix plots
  federated.py          client partitioning, fedavg, local steps, server loop
  experiments.py        paired scenarios, experiment tables, report plots
tests/
  conftest.py           synthetic Framingham-shaped fixture
  test_data.py
  test_evaluate.py
  test_config.py
  test_federated.py     Task 2 tests: FedAvg math, partitioning, seed determinism
  test_main.py
  test_app.py
data/README.md          dataset instructions (raw CSV is gitignored)
archive/                original prototype, kept for reference only
results/
  baseline/             convergence curve, comparison JSON, metrics, plots
team_plan.md            four-person task plan
reports/
  task1_report.md       Task 1 documentation
  task2_report.md       Task 2 baseline report by Pratik Rakhecha
```

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project was verified with Python 3.14 and the versions in
`requirements.txt`.

## Data and evaluation API

Tasks 2–4 should import the frozen interfaces instead of reimplementing data
handling or metrics:

```python
from src.config import Config
from src.data import load_data, prepare_data
from src.evaluate import evaluate_model, save_metrics

config = Config()
bundle = load_data(
    config.data_path,
    target_col=config.target_col,
    test_size=config.test_size,
    val_size=config.val_size,
    random_state=config.seed,
)

# bundle.X_train, bundle.X_val, bundle.X_test are scaled DataFrames.
# bundle.y_train, bundle.y_val, bundle.y_test are aligned Series.
# bundle.scaler, bundle.preprocessor, bundle.feature_names, bundle.metadata
# are available for inspection and saving.

metrics = evaluate_model(model, bundle.X_val, bundle.y_val)
save_metrics(metrics, config.results_dir / "metrics.json")
```

For an uploaded DataFrame (for example in Streamlit), call
`prepare_data(frame, ...)` instead of `load_data(path, ...)`.

`src.data` guarantees:

- a stratified, seeded train/validation/test split
- binary 0/1 target values
- median/mode imputation and one-hot encoding fitted on train only
- `StandardScaler` fitted on train only
- aligned, finite feature matrices with no missing values

## Dataset

Place the Framingham CSV at `data/framingham.csv`. See
[`data/README.md`](data/README.md) for the expected columns and preprocessing
details.

The raw CSV is **not** committed to the repository.

## Run the clean FedAvg baseline (Task 2)

Run the clean federated learning baseline and centralized reference model with one command:

```bash
python main.py --baseline
```

Or execute the federated module directly:

```bash
python -m src.federated
```

Useful options:

```text
--data PATH             explicit CSV path
--protocol {fedavg,one_shot}
                        FL protocol: true FedAvg (default) or one-shot
--n-clients INT         number of clients (default: 20)
--epochs INT            communication rounds (default: 50)
--lr FLOAT              client learning rate (default: 0.2)
--seed INT              random seed (default: 42)
--results-dir PATH      output directory (default: results/baseline)
```

Outputs written to `results/baseline/`:
- `fedavg_test_metrics.json` & `fedavg_val_metrics.json`
- `centralized_test_metrics.json` & `centralized_val_metrics.json`
- `fedavg_test_confusion_matrix.png` & `centralized_test_confusion_matrix.png`
- `convergence.csv` & `convergence.png`
- `comparison.json` (acceptance criteria tolerance check)
- `baseline_config.json`

## Run the foundation check

```bash
python main.py
```

This creates:

- `results/run_config.json`
- `results/data_summary.json`

To also save the prepared splits and run a centralized logistic-regression
sanity check on the validation split:

```bash
python main.py --save-splits --smoke-train
```

The smoke check is **not** the Task 2 FedAvg baseline. It only verifies that
the data and evaluation pipeline works.

To run a final held-out test evaluation (use sparingly):

```bash
python main.py --smoke-train --evaluate-test
```

Useful options:

```text
--data PATH             explicit CSV path
--target COLUMN         target column name (default: TenYearCHD)
--seed INT              split/model seed (default: 42)
--test-size FLOAT       test fraction (default: 0.2)
--val-size FLOAT        validation fraction (default: 0.1)
--results-dir PATH      output directory (default: results)
--save-splits           save prepared train/val/test CSVs
--smoke-train           run the centralized sanity check
--evaluate-test         also evaluate that smoke model on test
```

## Run the Task 4 experiment matrix

With the team's CSV at `data/framingham.csv`, run:

```bash
python run_experiments.py
```

The core matrix uses seeds 42–44, 50 rounds, and random noise at malicious-client
fractions 0%, 10%, 20%, 30%, and 50%. Each seed uses one shared data split and
client partition. The 0% case checks defense overhead on honest clients.

```bash
python run_experiments.py --matrix extended --results-dir results/extended
```

The extended matrix runs random noise, sign flip, scaling, label flip, and
collusion at a 20% malicious-client fraction. Use `--validation-only` during
exploration and `--help` for seed, strength, and training overrides.

- `summary.csv` records validation metrics by round and a separate final test row per run.
- `aggregate.csv` reports mean and sample standard deviation across seeds.
- `paired_comparisons.csv` reports clean minus attacked and defended minus attacked scores per seed.
- `plots/` contains convergence, final metrics versus malicious fraction, and defense gain.
- `experiment_config.json` records the settings and seeds for reruns.

See [Task 4 report notes](reports/task4_report.md) for the evaluation protocol.

## Run the app

```bash
streamlit run app.py
```

The app currently provides:

- CSV upload and default `data/framingham.csv` loading
- configurable seed, test fraction, and validation fraction
- split sizes and positive rates
- prepared training-feature preview and metadata
- a validation-only centralized smoke check

The Attack tab runs an interactive, validation-only comparison and exports its
metrics as CSV. Use the CLI for the full multi-seed results and final test metrics.

The old `streamlit.py` prototype was moved to
[`archive/streamlit_prototype.py`](archive/streamlit_prototype.py). It is not
maintained and contains known bugs; use `app.py`.

## Tests

```bash
pytest
```

The test suite does not require the real Framingham CSV. It generates a
synthetic dataset with the same schema and verifies:

- split sizes, disjointness, and seed determinism
- train-only preprocessing (validation/test values cannot change the scaler)
- missing-value imputation and categorical one-hot encoding
- target validation and missing-target handling
- metric correctness, JSON round-tripping, and plot creation
- end-to-end CLI execution
- Streamlit app-shell rendering

## Reproducibility rules

- Use `Config` (`src/config.py`) for every experiment.
- Use `np.random.default_rng(seed)` for new random operations.
- Keep the test split untouched until final evaluation.
- The defense must use the validation split, not the test split.
- Report mean ± standard deviation across seeds, not a single lucky run.
- No metric offsets such as `+0.2` and no R² on binary labels.

## Team

See [`team_plan.md`](team_plan.md) for the full four-person plan.

| Member | Primary workstream |
|---|---|
| Kunal Khandelwal | Foundation, data pipeline, evaluation harness (Task 1) |
| Pratik Rakhecha | Clean FedAvg baseline + centralized reference (Task 2) |
| Kunal V Kurundwadkar | Defense mechanism + validation protocol (Task 3) |
| Dewansh Khandelwal | Experiment orchestration and comparisons (Task 4, after the task swap) |
