# Experiment 04: B4 with test-set checkpoint selection

## Objective

Run the B4 hyperparameters found in the previous initial-gate and
impact-loss-weight sweep, while selecting the best epoch directly on the test
split as requested. This experiment does not use top-K filtering.

## Fixed configuration

```text
model                 = B4
seed                  = 2026
selection_split       = test
selection_metric      = weighted_f1
epochs                = 30
patience              = 7
gate_lr               = 0.0003
initial_gate          = 0.90
lambda_identity       = 0.001
lambda_impact         = 0.10
impact_loss           = soft_mse
impact_min_margin     = 0
gate_modalities       = text audio visual
backbone_frozen       = true
classifier_frozen     = true
```

The earlier `77.1256` value was a validation score, so it is not the target
metric of this protocol. Because test labels are used to select the epoch, the
reported test score is exploratory and optimistically biased; it must not be
presented as an untouched final-test estimate.

## Run

```bash
bash scripts/run_experiment_04.sh
```

To repeat over multiple seeds without changing the other settings:

```bash
E04_SEEDS="2024 2025 2026 2027 2028" bash scripts/run_experiment_04.sh
```

Primary output:

```text
outputs/full_run/Experiment_04_test_selection/E3/model_comparison.csv
```
