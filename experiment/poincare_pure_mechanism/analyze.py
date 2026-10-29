"""Mechanism analysis for pure-feature Emotion-Wheel experiments."""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, silhouette_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader import DialogueDataset, make_loaders
from model import MODALITIES, Transformer_Based_Model
from tical import emotion_wheel_angles, circular_class_distance_matrix
from train import build_parser, make_criterion


MODALITY_NAMES = {"t": "text", "a": "audio", "v": "visual"}
CLASS_NAMES = {
    "IEMOCAP": ("happy", "sad", "neutral", "angry", "excited", "frustrated"),
    "MELD": ("neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-path")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-batches", type=int, default=8)
    parser.add_argument("--amplification-pairs", type=int, default=50000)
    parser.add_argument("--silhouette-samples", type=int, default=1000)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def resolve_device(args):
    if args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu")
    return torch.device("cuda", args.gpu_id)


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def rankdata(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def pearson(first, second):
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    first, second = first[valid], second[valid]
    if first.size < 2:
        return float("nan")
    first, second = first - first.mean(), second - second.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(np.dot(first, second) / denominator) if denominator > 1e-12 else float("nan")


def spearman(first, second):
    return pearson(rankdata(first), rankdata(second))


def infer_id(checkpoint):
    args, config = checkpoint.get("args", {}), checkpoint["model_config"]
    if not config.get("use_emotion_wheel", False):
        return "B0"
    proto = float(args.get("lambda_wheel_proto", 0.0))
    cpcc = float(args.get("lambda_wheel_cpcc", 0.0))
    if proto == 0.0 and cpcc == 0.0:
        return "P-0"
    return {
        "euclidean": "E-E",
        "ball_euclidean": "B-E",
        "spherical": "S-S",
        "poincare": "B-P",
    }[config.get("wheel_geometry", "poincare")]


def discover(path, device):
    checkpoints = [path] if path.is_file() else sorted(path.rglob("best_checkpoint.pt"))
    found = {}
    for checkpoint_path in checkpoints:
        checkpoint = load_checkpoint(checkpoint_path, device)
        experiment_id = infer_id(checkpoint)
        seed = int(checkpoint.get("args", {}).get("seed", 2024))
        key = (experiment_id, seed)
        if key in found:
            previous = found[key]
            keep, skip = ((checkpoint_path, previous)
                          if checkpoint_path.stat().st_mtime >= previous.stat().st_mtime
                          else (previous, checkpoint_path))
            found[key] = keep
            print("Duplicate {} seed {}; using newest {}, skipping {}".format(
                experiment_id, seed, keep, skip), flush=True)
            continue
        found[key] = checkpoint_path
    if not found:
        raise ValueError("no best_checkpoint.pt found under {}".format(path))
    return found


def load_run(path, device, cli_args):
    checkpoint = load_checkpoint(path, device)
    config = dict(checkpoint["model_config"])
    model = Transformer_Based_Model(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.capture_pure = True
    model.set_tical_epoch(checkpoint.get("epoch", 0))
    dataset = DialogueDataset(
        config["dataset"], cli_args.feature_path or checkpoint.get("feature_path"))
    loaders = make_loaders(
        dataset, checkpoint["split_ids"], cli_args.batch_size,
        int(checkpoint.get("args", {}).get("seed", 2024)), num_workers=0,
        pin_memory=False)
    return checkpoint, model, loaders


def collect(model, loader, device):
    chunks = {"labels": [], "predictions": [], "probabilities": [],
              "pure": [], "projected": []}
    sample_ids = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            text, visual, audio, speakers, mask, labels = [item.to(device) for item in batch[:6]]
            valid = mask.bool()
            lengths = valid.sum(dim=1).tolist()
            output = model(text, visual, audio, mask, speakers.transpose(0, 1), lengths)
            chunks["labels"].append(labels[valid].cpu())
            chunks["predictions"].append(output["logits"][valid].argmax(-1).cpu())
            chunks["probabilities"].append(output["prob"][valid].float().cpu())
            chunks["pure"].append(output["pure"][valid].float().cpu())
            if output.get("tical") is not None:
                chunks["projected"].append(torch.stack([
                    output["tical"]["projected"][name][valid]
                    for name in MODALITIES], dim=1).float().cpu())
            for dialogue_id, length in zip(batch[6], lengths):
                sample_ids.extend((str(dialogue_id), index) for index in range(int(length)))
    result = {key: torch.cat(value).numpy() for key, value in chunks.items() if value}
    result["sample_ids"] = sample_ids
    return result


def unit(values):
    return values / np.clip(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12, None)


def wheel_tier(first, second, n_classes, class_distances):
    steps = int(round(float(class_distances[first, second]) * n_classes / 2.0))
    maximum = n_classes // 2
    if steps == 0:
        return "same"
    if steps == 1:
        return "adjacent"
    if steps == maximum:
        return "far"
    return "middle"


def pure_geometry_rows(experiment_id, seed, train, test, class_distances,
                       silhouette_samples):
    labels_train, labels_test = train["labels"], test["labels"]
    n_classes = class_distances.shape[0]
    semantic_pairs = class_distances[np.triu_indices(n_classes, 1)]
    rows, per_class = [], []
    for index, name in enumerate(MODALITIES):
        train_features = train["pure"][:, index]
        test_features = test["pure"][:, index]
        centroids = np.stack([
            train_features[labels_train == class_index].mean(axis=0)
            for class_index in range(n_classes)])
        distances = 1.0 - unit(test_features) @ unit(centroids).T
        correct = distances[np.arange(labels_test.size), labels_test]
        masked = distances.copy()
        masked[np.arange(labels_test.size), labels_test] = np.inf
        wrong = masked.min(axis=1)
        normalized_margin = (wrong - correct) / np.clip(wrong + correct, 1e-12, None)

        global_centroid = train_features.mean(axis=0)
        within = np.square(
            train_features - centroids[labels_train]).sum(axis=1).mean()
        between = sum(
            np.sum(labels_train == class_index)
            * np.square(centroids[class_index] - global_centroid).sum()
            for class_index in range(n_classes)) / labels_train.size

        normalized = unit(test_features)
        similarity = normalized @ normalized.T
        np.fill_diagonal(similarity, -np.inf)
        knn_prediction = np.empty(labels_test.size, dtype=np.int64)
        neighbors = np.argpartition(-similarity, kth=4, axis=1)[:, :5]
        for row, indices in enumerate(neighbors):
            counts = np.bincount(labels_test[indices], minlength=n_classes)
            knn_prediction[row] = counts.argmax()
        selected = np.arange(labels_test.size)
        if selected.size > silhouette_samples:
            selected = np.random.default_rng(seed).choice(
                selected, silhouette_samples, replace=False)
        silhouette = silhouette_score(
            normalized[selected], labels_test[selected], metric="cosine")

        centroid_distance = 1.0 - unit(centroids) @ unit(centroids).T
        centroid_pairs = centroid_distance[np.triu_indices(n_classes, 1)]
        rows.append({
            "id": experiment_id, "seed": seed, "modality": name,
            "within_scatter": float(within), "between_scatter": float(between),
            "fisher_ratio": float(between / max(within, 1e-12)),
            "centroid_margin_mean": float(normalized_margin.mean()),
            "negative_margin_rate": float(np.mean(normalized_margin < 0)),
            "knn5_accuracy": float(np.mean(knn_prediction == labels_test)),
            "silhouette_cosine": float(silhouette),
            "centroid_wheel_spearman": spearman(centroid_pairs, semantic_pairs),
        })
        for class_index in range(n_classes):
            mask = labels_test == class_index
            per_class.append({
                "id": experiment_id, "seed": seed, "modality": name,
                "class": class_index, "count": int(mask.sum()),
                "margin_mean": float(normalized_margin[mask].mean()),
                "negative_margin_rate": float(np.mean(normalized_margin[mask] < 0)),
            })
    return rows, per_class


def gradient_alignment(checkpoint, model, loader, device, max_batches,
                       class_distances, experiment_id, seed):
    if experiment_id in ("B0", "P-0") or max_batches == 0:
        return [], []
    saved_args = build_parser().parse_args([])
    for key, value in checkpoint.get("args", {}).items():
        setattr(saved_args, key, value)
    criterion = make_criterion(saved_args, device)
    model.eval()
    detail = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        text, visual, audio, speakers, mask, labels = [item.to(device) for item in batch[:6]]
        valid = mask.bool()
        lengths = valid.sum(dim=1).tolist()
        output = model(text, visual, audio, mask, speakers.transpose(0, 1), lengths)
        _, parts, _ = criterion(output, labels, valid)
        wheel = parts["weighted_wheel_proto"] + parts["weighted_wheel_cpcc"]
        pure = tuple(output["pure_modalities"][name] for name in MODALITIES)
        task_grad = torch.autograd.grad(parts["sdt"], pure, retain_graph=True)
        wheel_grad = torch.autograd.grad(wheel, pure)
        predictions = output["logits"].argmax(-1)
        confidence = output["prob"].amax(-1)
        for modality_index, name in enumerate(MODALITIES):
            first, second = task_grad[modality_index][valid], wheel_grad[modality_index][valid]
            dot = (first * second).sum(-1)
            norm_first, norm_second = first.norm(dim=-1), second.norm(dim=-1)
            cosine = dot / (norm_first * norm_second).clamp_min(1e-12)
            for row in range(cosine.numel()):
                target = int(labels[valid][row])
                prediction = int(predictions[valid][row])
                conf = float(confidence[valid][row])
                detail.append({
                    "id": experiment_id, "seed": seed, "batch": batch_index,
                    "modality": name, "cosine": float(cosine[row]),
                    "task_grad_norm": float(norm_first[row]),
                    "wheel_grad_norm": float(norm_second[row]),
                    "correct": int(target == prediction),
                    "confidence_group": "low" if conf < 0.6 else ("medium" if conf < 0.8 else "high"),
                    "error_tier": wheel_tier(target, prediction, class_distances.shape[0], class_distances),
                })
    grouped = defaultdict(list)
    for row in detail:
        grouped[(row["id"], row["seed"], row["modality"],
                 row["correct"], row["confidence_group"], row["error_tier"])].append(row)
    summary = []
    for key, values in sorted(grouped.items()):
        cosines = np.asarray([value["cosine"] for value in values])
        summary.append({
            "id": key[0], "seed": key[1], "modality": key[2],
            "correct": key[3], "confidence_group": key[4], "error_tier": key[5],
            "n": len(values), "cosine_mean": float(cosines.mean()),
            "positive_fraction": float(np.mean(cosines > 0)),
            "task_grad_norm_mean": float(np.mean([value["task_grad_norm"] for value in values])),
            "wheel_grad_norm_mean": float(np.mean([value["wheel_grad_norm"] for value in values])),
        })
    return detail, summary


def error_transitions(baseline, variant, reference_id, experiment_id, seed,
                      class_distances):
    if baseline["sample_ids"] != variant["sample_ids"] or not np.array_equal(
            baseline["labels"], variant["labels"]):
        raise ValueError("test utterance order differs between B0 and {}".format(experiment_id))
    labels = baseline["labels"]
    old, new = baseline["predictions"], variant["predictions"]
    rows = []
    for tier in ("same", "adjacent", "middle", "far"):
        selected = np.asarray([
            wheel_tier(int(labels[i]), int(old[i]), class_distances.shape[0], class_distances) == tier
            for i in range(labels.size)])
        old_wrong = selected & (old != labels)
        old_correct = selected & (old == labels)
        rows.append({
            "reference_id": reference_id, "id": experiment_id, "seed": seed,
            "baseline_error_tier": tier,
            "n": int(selected.sum()), "baseline_wrong": int(old_wrong.sum()),
            "wrong_to_correct": int(np.sum(old_wrong & (new == labels))),
            "correction_rate": (
                float(np.mean(new[old_wrong] == labels[old_wrong]))
                if old_wrong.any() else None),
            "baseline_correct": int(old_correct.sum()),
            "correct_to_wrong": int(np.sum(old_correct & (new != labels))),
            "harm_rate": (
                float(np.mean(new[old_correct] != labels[old_correct]))
                if old_correct.any() else None),
        })
    return rows


def distance_amplification(data, seed, class_distances, pair_count):
    if "projected" not in data:
        return [], []
    rng = np.random.default_rng(seed)
    labels, projected = data["labels"], data["projected"]
    n = labels.size
    first = rng.integers(0, n, size=pair_count)
    second = rng.integers(0, n, size=pair_count)
    equal = first == second
    second[equal] = (second[equal] + 1) % n
    detail, summary = [], []
    for modality_index, name in enumerate(MODALITIES):
        x, y = projected[first, modality_index], projected[second, modality_index]
        euclidean = np.linalg.norm(x - y, axis=-1)
        x_sq, y_sq = np.square(x).sum(-1), np.square(y).sum(-1)
        denominator = np.clip((1.0 - x_sq) * (1.0 - y_sq), 1e-5, None)
        argument = np.clip(1.0 + 2.0 * np.square(x - y).sum(-1) / denominator, 1.0 + 1e-5, None)
        poincare = np.arccosh(argument)
        amplification = poincare / np.clip(euclidean, 1e-8, None)
        mean_radius = 0.5 * (np.sqrt(x_sq) + np.sqrt(y_sq))
        radius_bin = np.minimum((mean_radius * 5).astype(int), 4)
        tiers = np.asarray([
            wheel_tier(int(labels[a]), int(labels[b]), class_distances.shape[0], class_distances)
            for a, b in zip(first, second)])
        for bin_index in range(5):
            for tier in ("same", "adjacent", "middle", "far"):
                mask = (radius_bin == bin_index) & (tiers == tier)
                if not mask.any():
                    continue
                summary.append({
                    "seed": seed, "modality": name,
                    "radius_bin": "{:.1f}-{:.1f}".format(bin_index / 5, (bin_index + 1) / 5),
                    "wheel_tier": tier, "n": int(mask.sum()),
                    "radius_mean": float(mean_radius[mask].mean()),
                    "euclidean_mean": float(euclidean[mask].mean()),
                    "poincare_mean": float(poincare[mask].mean()),
                    "amplification_mean": float(amplification[mask].mean()),
                })
        detail.append({
            "modality": name,
            "radius_amplification_spearman": spearman(mean_radius, amplification),
            "amplification_mean": float(amplification.mean()),
            "amplification_q95": float(np.quantile(amplification, 0.95)),
        })
    return detail, summary


def performance_row(experiment_id, seed, checkpoint, test):
    labels, predictions = test["labels"], test["predictions"]
    return {
        "id": experiment_id, "seed": seed, "best_epoch": int(checkpoint.get("epoch", 0)),
        "accuracy": 100.0 * accuracy_score(labels, predictions),
        "weighted_f1": 100.0 * f1_score(labels, predictions, average="weighted", zero_division=0),
        "macro_f1": 100.0 * f1_score(labels, predictions, average="macro", zero_division=0),
    }


def causal_deltas(performance):
    by_key = {(row["id"], row["seed"]): row for row in performance}
    comparisons = (
        ("projector_control", "P-0", "B0"),
        ("wheel_euclidean", "E-E", "P-0"),
        ("ball_map", "B-E", "E-E"),
        ("hyperbolic_metric", "B-P", "B-E"),
        ("poincare_vs_sphere", "B-P", "S-S"),
        ("poincare_total", "B-P", "P-0"),
        ("poincare_vs_raw_sdt", "B-P", "B0"),
    )
    rows = []
    for seed in sorted({row["seed"] for row in performance}):
        for hypothesis, left, right in comparisons:
            if (left, seed) not in by_key or (right, seed) not in by_key:
                continue
            rows.append({
                "hypothesis": hypothesis, "comparison": left + " - " + right,
                "seed": seed,
                "delta_weighted_f1": (
                    by_key[(left, seed)]["weighted_f1"]
                    - by_key[(right, seed)]["weighted_f1"]),
                "delta_macro_f1": (
                    by_key[(left, seed)]["macro_f1"]
                    - by_key[(right, seed)]["macro_f1"]),
                "delta_accuracy": (
                    by_key[(left, seed)]["accuracy"]
                    - by_key[(right, seed)]["accuracy"]),
            })
    return rows


def render_plots(output_dir, performance, pure_rows, amplification, no_plots):
    if no_plots:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ids = ["B0", "P-0", "E-E", "B-E", "S-S", "B-P"]
    means = {key: np.mean([row["weighted_f1"] for row in performance if row["id"] == key])
             for key in ids if any(row["id"] == key for row in performance)}
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(list(means), list(means.values()))
    axis.set_ylabel("Test weighted F1")
    axis.set_title("Map/metric ablation")
    axis.set_ylim(min(means.values()) - 1.0, max(means.values()) + 0.5)
    figure.tight_layout(); figure.savefig(output_dir / "performance.png", dpi=180); plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, modality in zip(axes, MODALITIES):
        selected = [row for row in pure_rows if row["modality"] == modality]
        values = defaultdict(list)
        for row in selected:
            values[row["id"]].append(row["fisher_ratio"])
        axis.bar(list(values), [np.mean(value) for value in values.values()])
        axis.set_title(MODALITY_NAMES[modality]); axis.set_ylabel("Pure-feature Fisher ratio")
    figure.tight_layout(); figure.savefig(output_dir / "pure_geometry.png", dpi=180); plt.close(figure)

    if amplification:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for modality in MODALITIES:
            selected = [row for row in amplification if row["modality"] == modality]
            grouped = defaultdict(list)
            for row in selected:
                grouped[row["radius_bin"]].append(row["amplification_mean"])
            axis.plot(list(grouped), [np.mean(value) for value in grouped.values()], marker="o", label=MODALITY_NAMES[modality])
        axis.set_xlabel("Mean ball-radius bin"); axis.set_ylabel("d_P / d_E")
        axis.set_title("Hyperbolic distance amplification"); axis.legend()
        figure.tight_layout(); figure.savefig(output_dir / "distance_amplification.png", dpi=180); plt.close(figure)


def write_report(path, performance, deltas, pure_rows, gradient_summary,
                 transitions, amplification_detail):
    lines = ["# Poincare pure-feature mechanism analysis", "",
             "## Performance: map versus metric", "",
             "| ID | Runs | Weighted F1 mean | Std |", "|---|---:|---:|---:|"]
    for experiment_id in ("B0", "P-0", "E-E", "B-E", "S-S", "B-P"):
        values = [row["weighted_f1"] for row in performance if row["id"] == experiment_id]
        if values:
            lines.append("| {} | {} | {:.4f} | {:.4f} |".format(
                experiment_id, len(values), np.mean(values), np.std(values)))
    lines.extend(["", "Interpretation: `B-E - E-E` isolates the bounded ball map; "
                  "`B-P - B-E` isolates the Poincare metric.", "",
                  "| Effect | Mean delta weighted F1 |", "|---|---:|"])
    for hypothesis in ("ball_map", "hyperbolic_metric", "poincare_vs_sphere", "poincare_total"):
        values = [row["delta_weighted_f1"] for row in deltas
                  if row["hypothesis"] == hypothesis]
        if values:
            lines.append("| {} | {:+.4f} |".format(hypothesis, np.mean(values)))
    lines.extend(["",
                  "## Pure H_mm geometry", "",
                  "| ID | Modality | Fisher | kNN@5 | Margin | Wheel Spearman |",
                  "|---|---|---:|---:|---:|---:|"])
    for experiment_id in ("B0", "P-0", "E-E", "B-E", "S-S", "B-P"):
        for modality in MODALITIES:
            rows = [row for row in pure_rows if row["id"] == experiment_id and row["modality"] == modality]
            if rows:
                lines.append("| {} | {} | {:.5f} | {:.4f} | {:.4f} | {:.4f} |".format(
                    experiment_id, MODALITY_NAMES[modality],
                    np.mean([row["fisher_ratio"] for row in rows]),
                    np.mean([row["knn5_accuracy"] for row in rows]),
                    np.mean([row["centroid_margin_mean"] for row in rows]),
                    np.mean([row["centroid_wheel_spearman"] for row in rows])))
    lines.extend(["", "## Gradient alignment at H_mm", "",
                  "See `gradient_alignment_summary.csv`. Positive cosine means the Wheel objective locally supports the SDT objective.", "",
                  "## Error transitions", "",
                  "See `error_transitions.csv` for Wrong->Correct and Correct->Wrong rates by semantic tier.", "",
                  "## Hyperbolic amplification", ""])
    for row in amplification_detail:
        lines.append("- {}: mean dP/dE={:.3f}, q95={:.3f}, Spearman(radius, amplification)={:.3f}.".format(
            MODALITY_NAMES[row["modality"]], row["amplification_mean"],
            row["amplification_q95"], row["radius_amplification_spearman"]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.gradient_batches < 0 or args.amplification_pairs < 1:
        raise ValueError("invalid analysis limits")
    device = resolve_device(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = discover(args.path.resolve(), device)

    performance, pure_rows, pure_class_rows = [], [], []
    gradient_detail, gradient_summary = [], []
    compact = {}
    amplification_detail, amplification_rows = [], []

    for (experiment_id, seed), checkpoint_path in sorted(runs.items()):
        print("Analysing {} seed {}".format(experiment_id, seed), flush=True)
        checkpoint, model, loaders = load_run(checkpoint_path, device, args)
        dataset_name = checkpoint["model_config"]["dataset"]
        n_classes = len(CLASS_NAMES[dataset_name])
        angles = emotion_wheel_angles(dataset_name, n_classes).numpy()
        class_distances = circular_class_distance_matrix(torch.from_numpy(angles)).numpy()
        train = collect(model, loaders["train"], device)
        test = collect(model, loaders["test"], device)
        performance.append(performance_row(experiment_id, seed, checkpoint, test))
        rows, classes = pure_geometry_rows(
            experiment_id, seed, train, test, class_distances,
            args.silhouette_samples)
        pure_rows.extend(rows); pure_class_rows.extend(classes)
        detail, summary = gradient_alignment(
            checkpoint, model, loaders["train"], device,
            args.gradient_batches, class_distances, experiment_id, seed)
        gradient_detail.extend(detail); gradient_summary.extend(summary)
        compact[(experiment_id, seed)] = {
            key: test[key] for key in ("labels", "predictions", "probabilities", "sample_ids")
        }
        if experiment_id == "B-P":
            detail_amp, rows_amp = distance_amplification(
                test, seed, class_distances, args.amplification_pairs)
            for row in detail_amp:
                row["seed"] = seed
            amplification_detail.extend(detail_amp); amplification_rows.extend(rows_amp)
        del model, train, test
        if device.type == "cuda":
            torch.cuda.empty_cache()

    transitions = []
    for (experiment_id, seed), variant in compact.items():
        reference_id = "P-0" if ("P-0", seed) in compact else "B0"
        if experiment_id in ("B0", "P-0") or (reference_id, seed) not in compact:
            continue
        n_classes = len(CLASS_NAMES["IEMOCAP"])
        class_distances = circular_class_distance_matrix(
            emotion_wheel_angles("IEMOCAP", n_classes)).numpy()
        transitions.extend(error_transitions(
            compact[(reference_id, seed)], variant, reference_id,
            experiment_id, seed, class_distances))

    deltas = causal_deltas(performance)

    write_csv(args.output_dir / "performance.csv", performance)
    write_csv(args.output_dir / "causal_deltas.csv", deltas)
    write_csv(args.output_dir / "pure_geometry.csv", pure_rows)
    write_csv(args.output_dir / "pure_geometry_per_class.csv", pure_class_rows)
    write_csv(args.output_dir / "gradient_alignment_samples.csv", gradient_detail)
    write_csv(args.output_dir / "gradient_alignment_summary.csv", gradient_summary)
    write_csv(args.output_dir / "error_transitions.csv", transitions)
    write_csv(args.output_dir / "distance_amplification.csv", amplification_rows)
    write_json(args.output_dir / "distance_amplification_summary.json", amplification_detail)
    write_report(args.output_dir / "REPORT.md", performance, deltas,
                 pure_rows, gradient_summary, transitions,
                 amplification_detail)
    render_plots(args.output_dir, performance, pure_rows,
                 amplification_rows, args.no_plots)
    print("Wrote analysis to {}".format(args.output_dir), flush=True)


if __name__ == "__main__":
    main()

