# Task 4 — Matched FedAvg experiments

**Workstream:** Dewansh Khandelwal (IIT2024173), following the Task 2/4 swap with Pratik Rakhecha.

## Experiment protocol

Each seed produces one stratified train, validation, and test split. All
settings start from zero parameters, use the same client partitions and local
updates, and run for the same number of communication rounds. A seeded
permutation selects the simulated malicious clients. The attacked and defended
runs use the same client IDs and random streams. Sample-weighted FedAvg is used
for the clean and undefended runs. The defended run uses Task 3's anomaly
scoring and robust aggregation on validation features.

The core matrix uses random noise, three seeds (42, 43, 44), 20 clients, 50
rounds, and malicious fractions of 0%, 10%, 20%, 30%, and 50%. At 0% the
defended run measures the effect of robust aggregation with only honest clients.
The extended matrix compares all five simulations at 20%. Label flipping
changes local training labels before the client's optimizer runs. Strength
controls the fraction flipped for that scenario. For random noise, strength
sets the standard deviation of each coefficient and the intercept uses ten
times that standard deviation, following Task 3's simulation.

## Evaluation and interpretation

The project proposal calls for convergence curves. Following the split protocol
in the team plan, curves use validation accuracy, F1 and ROC-AUC, with a separate
held-out test evaluation at the final round. The CSV also records precision and
recall. Use validation-only runs for exploratory parameter changes.

Report mean ± sample standard deviation across seeds. Within each seed,
calculate the attack drop as clean minus attacked and the defense gain as
defended minus attacked. Positive differences indicate degradation and recovery,
respectively. Check the zero-fraction defense result against the clean baseline
before claiming that the defense has little effect in the attack-free case.
The study is a simulation on one dataset, and benefits can differ by metric and
fraction.

Generate the core tables and figures with `python run_experiments.py`.
`results/summary.csv` contains the individual observations,
`results/aggregate.csv` contains the seed aggregates, and
`results/paired_comparisons.csv` contains the within-seed differences.

## Framingham results

The run used the public Framingham-derived CSV published by Kanda Sorn-In,
Wirapong Chansanam, and Pathamakorn Netayawijit (CC BY 4.0). It has 4,240 rows,
16 columns, and 645 missing values before preprocessing. The seed-42 clean
run reproduced the Task 2 held-out accuracy, F1, and ROC-AUC of 0.6686,
0.3510, and 0.6992. [Dataset record](https://doi.org/10.5281/zenodo.17971405)

Held-out test metrics for the core random-noise simulation, mean ± sample
standard deviation across three seeds:

| Malicious clients | Setting | Accuracy | F1 | ROC-AUC |
|---|---|---:|---:|---:|
| 0% | Clean FedAvg | 0.6702 ± 0.0038 | 0.3619 ± 0.0099 | 0.7040 ± 0.0096 |
| 0% | FEDDEF with honest clients | 0.6714 ± 0.0018 | 0.3627 ± 0.0098 | 0.7031 ± 0.0106 |
| 20% | Attacked FedAvg | 0.6509 ± 0.3220 | 0.2237 ± 0.0627 | 0.6819 ± 0.0035 |
| 20% | Attacked + FEDDEF | 0.8105 ± 0.0284 | 0.2508 ± 0.0668 | 0.6995 ± 0.0100 |
| 50% | Attacked FedAvg | 0.4430 ± 0.3622 | 0.2463 ± 0.0404 | 0.6171 ± 0.0241 |
| 50% | Attacked + FEDDEF | 0.4182 ± 0.3803 | 0.2442 ± 0.0430 | 0.6300 ± 0.0174 |

At 20%, random noise lowered both F1 and ROC-AUC relative to the clean
baseline. FEDDEF recovered about 1.76 ROC-AUC points and 2.71 F1 points
relative to the undefended run. At 50%, it recovered 1.29 ROC-AUC points
while accuracy and F1 fell slightly relative to the undefended run. With
only about 15% positive labels and substantial accuracy variation across
seeds, interpret accuracy alongside F1 and ROC-AUC.
