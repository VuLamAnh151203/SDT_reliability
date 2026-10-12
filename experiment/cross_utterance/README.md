# Frozen SDT cross-utterance impact analysis

This directory implements Phase 1 from
`SDT_Experiment1_Cross_Utterance_Impact_Plan.md`. It never trains or updates
the checkpoint. A single intervention replaces one raw source-utterance
feature vector in one modality, before SDT's Conv1D projection.

The model receives the complete dialogue. Every valid ordered source-target
pair is analyzed, except self-pairs (`source_idx == target_idx`).

## Requirements

- A fully trained checkpoint written by `SDT_new/train.py`.
- The matching IEMOCAP or MELD feature pickle.
- The packages in `SDT_new/requirements.txt`.
- Bash (Git Bash, WSL, or Linux/macOS shell).

Smoke checkpoints made with `--max-batches` are rejected by default.

## Run the four independent stages

Set shared environment variables once. Paths with spaces must remain quoted.

```bash
export CHECKPOINT="/absolute/path/to/best_checkpoint.pt"
export FEATURE_PATH="/absolute/path/to/iemocap_multimodal_features.pkl"
export RUN_DIR="SDT_new/experiment/cross_utterance/outputs/my_run"
export DEVICE="cuda"       # auto, cpu, or cuda
export GPU_ID="0"
export NUM_THREADS="0"     # 0 keeps PyTorch's default
export PROB_ATOL="2e-4"    # original-batch vs B=1 CUDA round-off only
export PROB_RTOL="2e-4"
```

Run Task A first. The remaining stages deliberately do not run automatically.

```bash
bash SDT_new/experiment/cross_utterance/run_task_a_baseline.sh
bash SDT_new/experiment/cross_utterance/run_task_b_zero_replacement.sh
bash SDT_new/experiment/cross_utterance/run_task_c_train_mean_replacement.sh
bash SDT_new/experiment/cross_utterance/run_analysis.sh
```

Task B and Task C require Task A's `experiment_manifest.json` and refuse to
run when the checkpoint, feature file, split, or seed differs. Intervention
tasks write one validated shard per dialogue and safely resume valid shards.

For an implementation-only smoke test:

```bash
export ALLOW_SMOKE_CHECKPOINT=1
export MAX_CONVERSATIONS=1
```

Never interpret a smoke-checkpoint result as a research result.

## Main outputs

```text
baseline/baseline_predictions.csv
baseline/baseline_metrics.json
feature_means/train_feature_means.npz
results/impact_pairs_zero.csv
results/impact_pairs_train_mean.csv
results/impact_pairs.csv
results/summary_by_modality.csv
results/summary_by_distance.csv
results/summary_by_speaker_relation.csv
results/masking_robustness.csv
results/ablation_classification_metrics.csv
results/test_classification_metrics.csv
results/changed_prediction_summary.csv
results/flip_rate_by_distance.csv
results/flip_rate_by_speaker_relation.csv
results/flip_rate_by_source_direction.csv
results/flip_rate_by_target_emotion.csv
results/flip_rate_by_source_emotion.csv
results/flip_rate_by_baseline_margin.csv
results/flip_class_transitions.csv
results/flip_target_susceptibility.csv
results/flip_robustness_zero_vs_mean.csv
results/top_changed_cases.csv
figures/impact_distribution_by_modality.png
figures/impact_vs_distance.png
figures/impact_by_speaker_relation.png
figures/zero_vs_mean_scatter.png
report.md
```

`run_analysis.sh` creates `ablation_classification_metrics.csv` automatically.
These ACC/F1 values are pair-weighted intervention metrics: every target is
repeated once per ablated source. They are not the ordinary utterance-level
test metrics from `baseline/baseline_metrics.json`.

It also creates `test_classification_metrics.csv` by deduplicating `pred_full`
on `(conversation_id, target_idx)`. Therefore, every test utterance contributes
exactly once. The script validates that repeated labels and baseline predictions
are consistent and that every target index in each conversation is present.

To calculate only the standard test metrics from an existing impact file:

```bash
python SDT_new/experiment/cross_utterance/compute_test_classification_metrics.py \
  --input /path/to/results/impact_pairs_train_mean.csv
```

To analyze the small subset of interventions that change the predicted class:

```bash
bash SDT_new/experiment/cross_utterance/run_prediction_change_analysis.sh
```

This joins impact rows to `baseline/baseline_predictions.csv`, derives baseline
top-1 confidence, top-1/top-2 margin, and entropy, and reports normalized flip
rates rather than raw counts alone. `top_changed_cases.csv` retains the 100
largest-JS-divergence flips per strategy and modality; override this with
`--top-k`.

To calculate the file separately from one completed strategy:

```bash
python SDT_new/experiment/cross_utterance/compute_ablation_classification_metrics.py \
  --input /path/to/results/impact_pairs_train_mean.csv
```

To include every completed strategy found under a run directory:

```bash
python SDT_new/experiment/cross_utterance/compute_ablation_classification_metrics.py \
  --run-dir /path/to/outputs/full_run
```

## Tests

```bash
python -m unittest \
  SDT_new/experiment/cross_utterance/test_cross_utterance_impact.py \
  SDT_new/experiment/cross_utterance/test_classification_metrics.py \
  SDT_new/experiment/cross_utterance/test_prediction_change_analysis.py \
  -v
```

The primary signed impact is

```text
CE_mask(target) - CE_full(target)
```

Positive impact means that replacing the source-modality feature increased the
target's true-class loss under that intervention. It is not a real-world causal
effect or an intrinsic measure of modality reliability.

