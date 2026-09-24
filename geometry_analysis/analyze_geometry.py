"""Post-hoc geometry diagnostics for trained SDT emotion-wheel checkpoints.

This script never trains or updates a model.  It loads a best checkpoint,
runs deterministic inference on a saved split, and measures the learned
Euclidean, spherical, or Poincare representation with its native distance.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

# Direct execution sets sys.path to geometry_analysis rather than SDT_new.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataloader import DialogueDataset, make_loaders
from geometry_analysis.geometry import geometry_distance
from model import MODALITIES, Transformer_Based_Model


CLASS_NAMES = {
    "IEMOCAP": ("happy", "sad", "neutral", "angry", "excited", "frustrated"),
    "MELD": ("neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"),
}
MODALITY_NAMES = {"t": "Text", "a": "Audio", "v": "Visual"}
TIER_ORDER = ("same", "adjacent", "middle", "far")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", help="best_checkpoint.pt or a directory containing checkpoints")
    parser.add_argument("--feature-path", help="override feature pickle path")
    parser.add_argument("--split", choices=("train", "valid", "test"),
                        default="test")
    parser.add_argument("--output-dir",
                        help="single-checkpoint output; default: beside checkpoint")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--max-utterances", type=int, default=2500,
                        help="class-balanced cap for pairwise, cluster, and k-NN analysis")
    parser.add_argument("--pair-cap-per-class-pair", type=int, default=20000,
                        help="cap used by class-pair-balanced correlations")
    parser.add_argument("--distance-chunk-size", type=int, default=128)
    parser.add_argument("--knn", type=int, nargs="+", default=(1, 3, 5, 10))
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def resolve_device(name, gpu_id):
    if name == "cpu" or (name == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    return torch.device("cuda", gpu_id)


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames or list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return value


def write_json(path, value):
    path.write_text(
        json.dumps(json_safe(value), indent=2, ensure_ascii=False),
        encoding="utf-8")


def summary_stats(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0, "mean": None, "std": None,
                "median": None, "q05": None, "q95": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
    }


def pearson(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    first, second = first[valid], second[valid]
    if first.size < 2:
        return None
    first, second = first - first.mean(), second - second.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-12:
        return None
    return float(np.dot(first, second) / denominator)


def rankdata(values):
    """Average ranks for ties, equivalent to scipy.stats.rankdata."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    first, second = first[valid], second[valid]
    if first.size < 2:
        return None
    return pearson(rankdata(first), rankdata(second))


def balanced_indices(labels, maximum, seed):
    labels = np.asarray(labels)
    if maximum <= 0 or labels.size <= maximum:
        return np.arange(labels.size)
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    base, remainder = divmod(maximum, len(classes))
    selected = []
    for position, class_index in enumerate(classes):
        indices = np.flatnonzero(labels == class_index)
        count = min(indices.size, base + int(position < remainder))
        selected.extend(rng.choice(indices, count, replace=False).tolist())
    # Fill unused capacity when a rare class had fewer than its allocation.
    if len(selected) < maximum:
        unused = np.setdiff1d(np.arange(labels.size), np.asarray(selected),
                              assume_unique=False)
        extra = rng.choice(
            unused, min(maximum - len(selected), unused.size), replace=False)
        selected.extend(extra.tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def collect_inference(model, loader, device, max_batches=0):
    chunks = {
        "labels": [], "predictions": [], "probabilities": [],
        **{"features_" + name: [] for name in MODALITIES},
        **{"prototype_distances_" + name: [] for name in MODALITIES},
        **{"wheel_predictions_" + name: [] for name in MODALITIES},
    }
    dialogue_ids, utterance_indices = [], []
    model.eval()
    with torch.no_grad():
        for batch_number, batch in enumerate(loader):
            if max_batches and batch_number >= max_batches:
                break
            text, visual, audio, speakers, mask, labels = [
                item.to(device) for item in batch[:6]
            ]
            valid = mask.bool()
            lengths = valid.sum(dim=1).tolist()
            output = model(
                text, visual, audio, mask,
                speakers.transpose(0, 1), lengths)
            tical = output.get("tical")
            if tical is None or not tical.get("wheel_enabled", False):
                raise ValueError(
                    "checkpoint has no emotion-wheel geometry to analyse")
            chunks["labels"].append(labels[valid].detach().cpu())
            chunks["predictions"].append(
                output["logits"][valid].argmax(dim=-1).detach().cpu())
            chunks["probabilities"].append(
                output["prob"][valid].detach().float().cpu())
            for name in MODALITIES:
                chunks["features_" + name].append(
                    tical["projected"][name][valid].detach().float().cpu())
                chunks["prototype_distances_" + name].append(
                    tical["wheel_distances"][name].detach().float().cpu())
                chunks["wheel_predictions_" + name].append(
                    tical["wheel_pseudo_labels"][name].detach().cpu())
            for dialogue_id, length in zip(batch[6], lengths):
                dialogue_ids.extend([dialogue_id] * length)
                utterance_indices.extend(range(length))
    if not chunks["labels"]:
        raise ValueError("selected split contains no utterances")
    result = {key: torch.cat(parts) for key, parts in chunks.items()}
    result["dialogue_ids"] = dialogue_ids
    result["utterance_indices"] = utterance_indices
    return result


def pairwise_distance_matrix(features, geometry, eps, device, chunk_size):
    features = features.to(device)
    count = features.size(0)
    matrix = np.empty((count, count), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, count, chunk_size):
            end = min(start + chunk_size, count)
            distance = geometry_distance(
                features[start:end, None, :], features[None, :, :],
                geometry, eps)
            matrix[start:end] = distance.detach().float().cpu().numpy()
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def wheel_steps(class_distances, n_classes):
    return np.rint(class_distances * n_classes / 2.0).astype(np.int64)


def tier_name(step, maximum_step):
    if step == 0:
        return "same"
    if step == 1:
        return "adjacent"
    if step == maximum_step:
        return "far"
    return "middle"


def class_balanced_pair_values(matrix, labels, semantic_matrix, cap, seed):
    rng = np.random.default_rng(seed)
    groups = []
    n_classes = semantic_matrix.shape[0]
    for first_class in range(n_classes):
        first = np.flatnonzero(labels == first_class)
        for second_class in range(first_class, n_classes):
            second = np.flatnonzero(labels == second_class)
            if first_class == second_class:
                row, column = np.triu_indices(first.size, 1)
                pair_values = matrix[first[row], first[column]]
            else:
                pair_values = matrix[np.ix_(first, second)].reshape(-1)
            if pair_values.size:
                groups.append((pair_values,
                               semantic_matrix[first_class, second_class]))
    equal_count = min(values.size for values, _ in groups)
    if cap > 0:
        equal_count = min(equal_count, cap)
    geometric, semantic = [], []
    for pair_values, semantic_value in groups:
        if pair_values.size > equal_count:
            chosen = rng.choice(pair_values.size, equal_count, replace=False)
            pair_values = pair_values[chosen]
        geometric.append(pair_values)
        semantic.append(np.full(
            pair_values.size, semantic_value, dtype=np.float32))
    return np.concatenate(geometric), np.concatenate(semantic)


def silhouette_values(matrix, labels):
    labels = np.asarray(labels)
    result = np.zeros(labels.size, dtype=np.float64)
    classes = np.unique(labels)
    for index in range(labels.size):
        same = np.flatnonzero(labels == labels[index])
        same = same[same != index]
        if not same.size:
            continue
        within = float(matrix[index, same].mean())
        between = min(float(matrix[index, labels == other].mean())
                      for other in classes if other != labels[index])
        denominator = max(within, between)
        result[index] = (between - within) / denominator if denominator > 0 else 0.0
    return result


def knn_rows(matrix, labels, modality, ks, n_classes):
    maximum = min(max(ks), labels.size - 1)
    if maximum < 1:
        return []
    work = matrix.copy()
    np.fill_diagonal(work, np.inf)
    neighbors = np.argpartition(work, maximum - 1, axis=1)[:, :maximum]
    local_distances = np.take_along_axis(work, neighbors, axis=1)
    order = np.argsort(local_distances, axis=1)
    neighbors = np.take_along_axis(neighbors, order, axis=1)
    rows = []
    for k in sorted(set(ks)):
        if k > maximum:
            continue
        prediction = np.asarray([
            np.bincount(labels[row[:k]], minlength=n_classes).argmax()
            for row in neighbors
        ])
        rows.append({
            "modality": modality,
            "k": k,
            "accuracy": 100.0 * accuracy_score(labels, prediction),
            "macro_f1": 100.0 * f1_score(
                labels, prediction, average="macro", zero_division=0),
        })
    return rows


def correlation_bundle(values, confidence, entropy, correctness, margin):
    targets = {
        "confidence": confidence,
        "negative_entropy": -entropy,
        "correctness": correctness,
        "prototype_margin": margin,
    }
    output = {}
    for name, target in targets.items():
        output[name] = {
            "pearson": pearson(values, target),
            "spearman": spearman(values, target),
        }
    return output


def confidence_bin_rows(confidence, radius, correctness, modality,
                        n_bins=10, radius_scale="ball_norm"):
    """Summarize Poincare radius in fixed confidence intervals."""
    confidence = np.asarray(confidence, dtype=np.float64)
    radius = np.asarray(radius, dtype=np.float64)
    correctness = np.asarray(correctness, dtype=np.float64)
    if not (confidence.shape == radius.shape == correctness.shape):
        raise ValueError("confidence, radius, and correctness shapes differ")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assignments = np.clip(np.digitize(confidence, edges, right=False) - 1,
                          0, n_bins - 1)
    rows = []
    for index in range(n_bins):
        mask = assignments == index
        if not mask.any():
            continue
        local_radius = radius[mask]
        rows.append({
            "modality": modality,
            "radius_scale": radius_scale,
            "bin": index,
            "confidence_low": float(edges[index]),
            "confidence_high": float(edges[index + 1]),
            "count": int(mask.sum()),
            "confidence_mean": float(confidence[mask].mean()),
            "radius_mean": float(local_radius.mean()),
            "radius_std": float(local_radius.std()),
            "radius_sem": float(
                local_radius.std() / math.sqrt(local_radius.size)),
            "radius_q25": float(np.quantile(local_radius, 0.25)),
            "radius_median": float(np.median(local_radius)),
            "radius_q75": float(np.quantile(local_radius, 0.75)),
            "accuracy": float(correctness[mask].mean()),
        })
    return rows


def infer_experiment_id(config, args):
    if not config.get("use_emotion_wheel", False):
        return "B0"
    prefix = {"euclidean": "E", "spherical": "S", "poincare": "P"}[
        config.get("wheel_geometry", "poincare")]
    has_proto = float(args.get("lambda_wheel_proto", 0.0)) > 0
    has_cpcc = float(args.get("lambda_wheel_cpcc", 0.0)) > 0
    suffix = "PC" if has_proto and has_cpcc else "P" if has_proto else "C"
    run_id = prefix + "-" + suffix
    dimension = int(config.get("hyperbolic_dim", 0))
    if config.get("wheel_radius_mode", "free") == "fixed":
        run_id += "-fixed-r{}".format(
            str(config.get("wheel_fixed_radius", 0.75)).replace(".", "p"))
    if config.get("wheel_zero_residual", False):
        run_id += "-zerores"
    if dimension != 16:
        run_id += "-d{}".format(dimension)
    return run_id


def analyze_checkpoint(checkpoint_path, cli_args, device):
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = dict(checkpoint["model_config"])
    saved_args = checkpoint.get("args", {})
    if not config.get("use_tical") or not config.get("use_emotion_wheel"):
        raise ValueError("not a wheel-enabled checkpoint")
    geometry = config.get("wheel_geometry", "poincare")
    dataset_name = config["dataset"]
    class_names = CLASS_NAMES[dataset_name]
    n_classes = len(class_names)
    feature_path = cli_args.feature_path or checkpoint.get("feature_path")
    dataset = DialogueDataset(dataset_name, feature_path)
    split_ids = checkpoint["split_ids"]
    if cli_args.split == "valid" and not split_ids.get("valid"):
        raise ValueError("checkpoint has no validation split")
    loaders = make_loaders(
        dataset, split_ids, cli_args.batch_size, cli_args.seed,
        num_workers=0, pin_memory=device.type == "cuda")
    model = Transformer_Based_Model(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.set_tical_epoch(checkpoint.get("epoch", 0))
    collected = collect_inference(
        model, loaders[cli_args.split], device, cli_args.max_batches)

    labels = collected["labels"].numpy()
    predictions = collected["predictions"].numpy()
    probabilities = collected["probabilities"].numpy()
    confidence = probabilities.max(axis=1)
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(axis=1)
    correctness = (predictions == labels).astype(np.float64)
    cm = confusion_matrix(labels, predictions, labels=np.arange(n_classes))
    precision, recall, final_f1, support = precision_recall_fscore_support(
        labels, predictions, labels=np.arange(n_classes), zero_division=0)
    class_distance = model.tical.wheel_class_distances.detach().cpu().numpy()
    prototypes = model.tical.wheel_prototypes.detach().float().cpu()
    hyp_eps = float(config.get("hyp_eps", 1e-5))

    selected = balanced_indices(
        labels, cli_args.max_utterances, cli_args.seed)
    selected_labels = labels[selected]
    maximum_step = n_classes // 2
    sample_rows = [{
        "dialogue_id": collected["dialogue_ids"][index],
        "utterance_index": collected["utterance_indices"][index],
        "label": int(labels[index]),
        "emotion": class_names[labels[index]],
        "prediction": int(predictions[index]),
        "predicted_emotion": class_names[predictions[index]],
        "correct": int(correctness[index]),
        "confidence": float(confidence[index]),
        "entropy": float(entropy[index]),
    } for index in range(labels.size)]
    per_class_rows, class_pair_rows, tier_rows, knn_output = [], [], [], []
    modality_reports, plot_data = {}, {}

    row_index, column_index = np.triu_indices(selected.size, 1)
    semantic_pair = class_distance[
        selected_labels[row_index], selected_labels[column_index]]
    step_pair = wheel_steps(semantic_pair, n_classes)

    for modality_position, name in enumerate(MODALITIES):
        features = collected["features_" + name]
        prototype_distances = collected[
            "prototype_distances_" + name].numpy()
        wheel_prediction = collected["wheel_predictions_" + name].numpy()
        correct_distance = prototype_distances[np.arange(labels.size), labels]
        wrong = prototype_distances.copy()
        wrong[np.arange(labels.size), labels] = np.inf
        nearest_wrong = wrong.min(axis=1)
        margin = nearest_wrong - correct_distance
        norms = features.norm(dim=-1).numpy()
        off_plane = features[:, 2:].norm(dim=-1).numpy()
        off_plane_ratio = off_plane / np.clip(norms, 1e-12, None)
        rho = (2.0 * np.arctanh(np.clip(norms, 0.0, 1.0 - hyp_eps))
               if geometry == "poincare" else None)

        matrix = pairwise_distance_matrix(
            features[selected], geometry, hyp_eps, device,
            cli_args.distance_chunk_size)
        geometric_pair = matrix[row_index, column_index]
        median_pair_distance = float(np.median(geometric_pair))
        scale = max(median_pair_distance, 1e-12)
        balanced_geometric, balanced_semantic = class_balanced_pair_values(
            matrix, selected_labels, class_distance,
            cli_args.pair_cap_per_class_pair,
            cli_args.seed + modality_position)
        silhouette = silhouette_values(matrix, selected_labels)
        silhouette_by_original = {
            int(original): float(value)
            for original, value in zip(selected, silhouette)
        }
        knn_output.extend(knn_rows(
            matrix, selected_labels, name, cli_args.knn, n_classes))

        tier_values = {}
        for step in range(maximum_step + 1):
            tier = tier_name(step, maximum_step)
            values = geometric_pair[step_pair == step]
            tier_values.setdefault(tier, []).append(values)
        merged_tiers = {
            tier: np.concatenate(parts) if parts else np.empty(0)
            for tier, parts in tier_values.items()
        }
        for tier in TIER_ORDER:
            if tier not in merged_tiers:
                continue
            stats = summary_stats(merged_tiers[tier])
            tier_rows.append({
                "modality": name, "tier": tier,
                **stats,
                "normalized_mean": (
                    stats["mean"] / scale if stats["mean"] is not None else None),
            })
        order_count = min(
            100000,
            *(merged_tiers[tier].size for tier in ("same", "adjacent", "far")))
        rng = np.random.default_rng(cli_args.seed + 100 + modality_position)
        if order_count:
            order_same = rng.choice(merged_tiers["same"], order_count, replace=True)
            order_adjacent = rng.choice(
                merged_tiers["adjacent"], order_count, replace=True)
            order_far = rng.choice(merged_tiers["far"], order_count, replace=True)
            ordering_accuracy = float(np.mean(
                (order_same < order_adjacent)
                & (order_adjacent < order_far)))
        else:
            ordering_accuracy = None

        class_mean_matrix = np.zeros((n_classes, n_classes), dtype=np.float64)
        for first_class in range(n_classes):
            first = np.flatnonzero(selected_labels == first_class)
            for second_class in range(first_class, n_classes):
                second = np.flatnonzero(selected_labels == second_class)
                if first_class == second_class:
                    local_row, local_column = np.triu_indices(first.size, 1)
                    values = matrix[first[local_row], first[local_column]]
                else:
                    values = matrix[np.ix_(first, second)].reshape(-1)
                stats = summary_stats(values)
                class_mean_matrix[first_class, second_class] = (
                    stats["mean"] if stats["mean"] is not None else 0.0)
                class_mean_matrix[second_class, first_class] = (
                    stats["mean"] if stats["mean"] is not None else 0.0)
                steps = int(round(
                    class_distance[first_class, second_class] * n_classes / 2.0))
                class_pair_rows.append({
                    "modality": name,
                    "class_a": first_class,
                    "emotion_a": class_names[first_class],
                    "class_b": second_class,
                    "emotion_b": class_names[second_class],
                    "wheel_steps": steps,
                    "wheel_distance": float(
                        class_distance[first_class, second_class]),
                    **stats,
                    "normalized_mean": (
                        stats["mean"] / scale
                        if stats["mean"] is not None else None),
                })

        prototype_matrix = pairwise_distance_matrix(
            prototypes, geometry, hyp_eps, device, n_classes)
        row_rates = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        prototype_pair_distance, negative_confusion = [], []
        for first_class in range(n_classes):
            for second_class in range(first_class + 1, n_classes):
                prototype_pair_distance.append(
                    prototype_matrix[first_class, second_class])
                negative_confusion.append(-0.5 * (
                    row_rates[first_class, second_class]
                    + row_rates[second_class, first_class]))

        for class_index in range(n_classes):
            class_mask = labels == class_index
            selected_class_mask = selected_labels == class_index
            competitors = [other for other in range(n_classes)
                           if other != class_index]
            nearest_class = min(
                competitors,
                key=lambda other: class_mean_matrix[class_index, other])
            class_rho = rho[class_mask] if rho is not None else np.empty(0)
            selected_silhouette = silhouette[selected_class_mask]
            per_class_rows.append({
                "modality": name,
                "class": class_index,
                "emotion": class_names[class_index],
                "count": int(class_mask.sum()),
                "final_precision": float(precision[class_index]),
                "final_recall": float(recall[class_index]),
                "final_f1": float(final_f1[class_index]),
                "wheel_accuracy": float(np.mean(
                    wheel_prediction[class_mask] == labels[class_mask])),
                "correct_distance_mean": float(correct_distance[class_mask].mean()),
                "wrong_distance_mean": float(nearest_wrong[class_mask].mean()),
                "margin_mean": float(margin[class_mask].mean()),
                "normalized_margin_mean": float(margin[class_mask].mean() / scale),
                "negative_margin_rate": float(np.mean(margin[class_mask] < 0)),
                "norm_mean": float(norms[class_mask].mean()),
                "rho_mean": float(class_rho.mean()) if class_rho.size else None,
                "off_plane_ratio_mean": float(off_plane_ratio[class_mask].mean()),
                "within_class_distance": float(
                    class_mean_matrix[class_index, class_index]),
                "silhouette_mean": float(selected_silhouette.mean()),
                "nearest_class": int(nearest_class),
                "nearest_emotion": class_names[nearest_class],
                "nearest_class_distance": float(
                    class_mean_matrix[class_index, nearest_class]),
            })

        for index, row in enumerate(sample_rows):
            prefix = name + "_"
            row.update({
                prefix + "wheel_prediction": int(wheel_prediction[index]),
                prefix + "wheel_correct": int(wheel_prediction[index] == labels[index]),
                prefix + "correct_distance": float(correct_distance[index]),
                prefix + "wrong_distance": float(nearest_wrong[index]),
                prefix + "prototype_margin": float(margin[index]),
                prefix + "normalized_margin": float(margin[index] / scale),
                prefix + "norm": float(norms[index]),
                prefix + "rho": float(rho[index]) if rho is not None else None,
                prefix + "off_plane_norm": float(off_plane[index]),
                prefix + "off_plane_ratio": float(off_plane_ratio[index]),
                prefix + "silhouette": silhouette_by_original.get(index),
            })

        radial_report = {
            "applicable": geometry == "poincare",
            "euclidean_norm": summary_stats(norms),
            "euclidean_norm_correct": summary_stats(
                norms[correctness.astype(bool)]),
            "euclidean_norm_incorrect": summary_stats(
                norms[~correctness.astype(bool)]),
            "norm_correlations": correlation_bundle(
                norms, confidence, entropy, correctness, margin),
            "hyperbolic_radius": summary_stats(rho) if rho is not None else None,
            "hyperbolic_radius_correct": (
                summary_stats(rho[correctness.astype(bool)])
                if rho is not None else None),
            "hyperbolic_radius_incorrect": (
                summary_stats(rho[~correctness.astype(bool)])
                if rho is not None else None),
            "boundary_fraction_norm_gt_0_90": (
                float(np.mean(norms > 0.90)) if geometry == "poincare" else None),
            "boundary_fraction_norm_gt_0_95": (
                float(np.mean(norms > 0.95)) if geometry == "poincare" else None),
            "boundary_fraction_norm_gt_0_99": (
                float(np.mean(norms > 0.99)) if geometry == "poincare" else None),
            "rho_correlations": (
                correlation_bundle(rho, confidence, entropy, correctness, margin)
                if rho is not None else None),
        }
        modality_reports[name] = {
            "title": MODALITY_NAMES[name],
            "prototype": {
                "wheel_accuracy": float(np.mean(wheel_prediction == labels)),
                "correct_distance": summary_stats(correct_distance),
                "nearest_wrong_distance": summary_stats(nearest_wrong),
                "margin": summary_stats(margin),
                "normalized_margin": summary_stats(margin / scale),
                "negative_margin_rate": float(np.mean(margin < 0)),
            },
            "semantic_distance": {
                "pair_weighted_pearson_cpcc": pearson(
                    geometric_pair, semantic_pair),
                "pair_weighted_spearman": spearman(
                    geometric_pair, semantic_pair),
                "class_pair_balanced_pearson_cpcc": pearson(
                    balanced_geometric, balanced_semantic),
                "class_pair_balanced_spearman": spearman(
                    balanced_geometric, balanced_semantic),
                "cpcc_loss_pair_weighted": (
                    1.0 - pearson(geometric_pair, semantic_pair)
                    if pearson(geometric_pair, semantic_pair) is not None else None),
                "median_pair_distance": median_pair_distance,
                "ordering_accuracy_same_adjacent_far": ordering_accuracy,
            },
            "radial": radial_report,
            "off_plane": {
                "norm": summary_stats(off_plane),
                "ratio": summary_stats(off_plane_ratio),
                "ratio_correlations": correlation_bundle(
                    off_plane_ratio, confidence, entropy, correctness, margin),
            },
            "cluster": {
                "silhouette": summary_stats(silhouette),
                "prototype_distance_vs_negative_confusion_pearson": pearson(
                    prototype_pair_distance, negative_confusion),
                "prototype_distance_vs_negative_confusion_spearman": spearman(
                    prototype_pair_distance, negative_confusion),
            },
            "knn": [row for row in knn_output if row["modality"] == name],
        }
        plot_data[name] = {
            "margin": margin,
            "norm_or_rho": norms,
            "ball_radius": norms,
            "hyperbolic_radius": rho,
            "confidence": confidence,
            "correctness": correctness,
            "tiers": merged_tiers,
            "class_mean_matrix": class_mean_matrix,
        }

    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "experiment_id": infer_experiment_id(config, saved_args),
        "dataset": dataset_name,
        "split": cli_args.split,
        "seed": int(saved_args.get("seed", cli_args.seed)),
        "epoch": int(checkpoint.get("epoch", 0)),
        "geometry": geometry,
        "dimension": int(config["hyperbolic_dim"]),
        "radius_mode": config.get("wheel_radius_mode", "free"),
        "fixed_radius": float(config.get("wheel_fixed_radius", 0.75)),
        "zero_residual": bool(config.get("wheel_zero_residual", False)),
        "lambda_proto": float(saved_args.get("lambda_wheel_proto", 0.0)),
        "lambda_cpcc": float(saved_args.get("lambda_wheel_cpcc", 0.0)),
        "n_utterances": int(labels.size),
        "n_pairwise_utterances": int(selected.size),
        "classification": {
            "accuracy": 100.0 * accuracy_score(labels, predictions),
            "weighted_f1": 100.0 * f1_score(
                labels, predictions, average="weighted", zero_division=0),
            "macro_f1": 100.0 * f1_score(
                labels, predictions, average="macro", zero_division=0),
        },
        "modalities": modality_reports,
    }

    output_dir = (Path(cli_args.output_dir).expanduser().resolve()
                  if cli_args.output_dir else
                  checkpoint_path.parent / ("geometry_diagnostics_" + cli_args.split))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "geometry_report.json", report)
    write_csv(output_dir / "sample_metrics.csv", sample_rows)
    write_csv(output_dir / "per_class_metrics.csv", per_class_rows)
    write_csv(output_dir / "class_pair_metrics.csv", class_pair_rows)
    write_csv(output_dir / "distance_tiers.csv", tier_rows)
    write_csv(output_dir / "knn_metrics.csv", knn_output)
    if geometry == "poincare":
        radius_confidence_bins = []
        for name in MODALITIES:
            radius_confidence_bins.extend(confidence_bin_rows(
                plot_data[name]["confidence"],
                plot_data[name]["ball_radius"],
                plot_data[name]["correctness"], name))
        write_csv(
            output_dir / "poincare_radius_confidence_bins.csv",
            radius_confidence_bins)
    confusion_rows = [
        {"true_class": class_index, "emotion": class_names[class_index],
         **{"pred_{}_{}".format(predicted_index, class_names[predicted_index]):
            int(cm[class_index, predicted_index])
            for predicted_index in range(n_classes)}}
        for class_index in range(n_classes)
    ]
    write_csv(output_dir / "confusion_matrix.csv", confusion_rows)
    write_markdown(output_dir / "geometry_report.md", report)
    if not cli_args.no_plots:
        render_plots(output_dir, plot_data, cm, class_names, geometry)
    print("Analysed {} -> {}".format(checkpoint_path, output_dir), flush=True)
    return report


def write_markdown(path, report):
    lines = [
        "# Geometry diagnostics", "",
        "- Experiment: `{}`".format(report["experiment_id"]),
        "- Geometry: `{}` ({}D); radius: `{}`; zero residual: `{}`".format(
            report["geometry"], report["dimension"],
            report["radius_mode"], report["zero_residual"]),
        "- Split: `{}`; utterances: {}; pairwise subset: {}".format(
            report["split"], report["n_utterances"],
            report["n_pairwise_utterances"]),
        "- Weighted F1: {:.3f}; macro F1: {:.3f}; accuracy: {:.3f}".format(
            report["classification"]["weighted_f1"],
            report["classification"]["macro_f1"],
            report["classification"]["accuracy"]),
        "", "| Modality | Wheel acc | Margin | Negative margin | CPCC | Spearman | Ordering | Silhouette |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in MODALITIES:
        item = report["modalities"][name]
        lines.append(
            "| {title} | {wheel:.2%} | {margin:.4f} | {negative:.2%} | "
            "{cpcc} | {spearman} | {ordering} | {silhouette:.4f} |".format(
                title=item["title"],
                wheel=item["prototype"]["wheel_accuracy"],
                margin=item["prototype"]["normalized_margin"]["mean"],
                negative=item["prototype"]["negative_margin_rate"],
                cpcc=_format_optional(item["semantic_distance"][
                    "class_pair_balanced_pearson_cpcc"]),
                spearman=_format_optional(item["semantic_distance"][
                    "class_pair_balanced_spearman"]),
                ordering=_format_optional(item["semantic_distance"][
                    "ordering_accuracy_same_adjacent_far"]),
                silhouette=item["cluster"]["silhouette"]["mean"]))
    lines.extend([
        "", "## Radius/norm versus final confidence", "",
        "| Modality | Measure | Mean | Correct | Incorrect | Pearson | Spearman |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for name in MODALITIES:
        item = report["modalities"][name]["radial"]
        if item["hyperbolic_radius"] is not None:
            measure = "Poincare ball radius"
            overall = item["euclidean_norm"]
            correct = item["euclidean_norm_correct"]
            incorrect = item["euclidean_norm_incorrect"]
            correlations = item["norm_correlations"]["confidence"]
        else:
            measure = "embedding norm"
            overall = item["euclidean_norm"]
            correct = item["euclidean_norm_correct"]
            incorrect = item["euclidean_norm_incorrect"]
            correlations = item["norm_correlations"]["confidence"]
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                MODALITY_NAMES[name], measure,
                _format_optional(overall["mean"]),
                _format_optional(correct["mean"]),
                _format_optional(incorrect["mean"]),
                _format_optional(correlations["pearson"]),
                _format_optional(correlations["spearman"])))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_optional(value):
    return "-" if value is None else "{:.4f}".format(value)


def render_plots(output_dir, plot_data, cm, class_names, geometry):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping plots", flush=True)
        return
    figure, axes = plt.subplots(4, 3, figsize=(18, 18))
    for column, name in enumerate(MODALITIES):
        data = plot_data[name]
        axes[0, column].hist(data["margin"], bins=45, alpha=0.8)
        axes[0, column].axvline(0.0, color="red", linestyle="--")
        axes[0, column].set_title(
            "{} prototype margin".format(MODALITY_NAMES[name]))
        axes[0, column].set_xlabel("nearest-wrong minus correct distance")

        tiers = [tier for tier in TIER_ORDER if tier in data["tiers"]]
        means = [float(data["tiers"][tier].mean()) for tier in tiers]
        errors = [float(data["tiers"][tier].std()) for tier in tiers]
        axes[1, column].bar(tiers, means, yerr=errors, alpha=0.8)
        axes[1, column].set_title("Pair distance by Wheel tier")
        axes[1, column].tick_params(axis="x", rotation=20)

        sample = np.linspace(
            0, data["confidence"].size - 1,
            min(1500, data["confidence"].size), dtype=int)
        axes[2, column].scatter(
            data["confidence"][sample], data["norm_or_rho"][sample],
            s=8, alpha=0.3)
        axes[2, column].set_xlabel("final confidence")
        axes[2, column].set_ylabel(
            "Poincare ball radius" if geometry == "poincare" else "embedding norm")
        if geometry == "poincare":
            axes[2, column].set_ylim(0.0, 1.0)
        radial_spearman = spearman(
            data["norm_or_rho"], data["confidence"])
        axes[2, column].set_title(
            "Radius/norm relation (Spearman={})".format(
                _format_optional(radial_spearman)))

        image = axes[3, column].imshow(
            data["class_mean_matrix"], cmap="viridis")
        axes[3, column].set_title("Mean class-pair distance")
        axes[3, column].set_xticks(range(len(class_names)), class_names, rotation=45,
                                  ha="right")
        axes[3, column].set_yticks(range(len(class_names)), class_names)
        figure.colorbar(image, ax=axes[3, column], fraction=0.046)
    figure.suptitle("{} geometry diagnostics".format(geometry), fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output_dir / "geometry_dashboard.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(cm, cmap="Blues")
    axis.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Final classifier confusion matrix")
    for row in range(cm.shape[0]):
        for column in range(cm.shape[1]):
            axis.text(column, row, str(cm[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)

    if geometry == "poincare":
        render_poincare_radius_confidence(output_dir, plot_data, plt)


def render_poincare_radius_confidence(output_dir, plot_data, plt):
    """Render a Poincare-only radius/confidence diagnostic figure."""
    figure, axes = plt.subplots(2, 3, figsize=(18, 10))
    for column, name in enumerate(MODALITIES):
        data = plot_data[name]
        confidence = np.asarray(data["confidence"], dtype=np.float64)
        # Coordinate radius in the Poincare ball.  Unlike geodesic radius
        # rho=2*atanh(||z||), this is bounded in [0, 1).
        radius = np.asarray(data["ball_radius"], dtype=np.float64)
        correctness = np.asarray(data["correctness"], dtype=bool)
        bins = confidence_bin_rows(
            confidence, radius, correctness.astype(float), name)

        top = axes[0, column]
        top.scatter(confidence, radius, s=9, alpha=0.18,
                    color="tab:blue", rasterized=True)
        bin_confidence = np.asarray(
            [row["confidence_mean"] for row in bins])
        bin_radius = np.asarray([row["radius_mean"] for row in bins])
        bin_sem = np.asarray([row["radius_sem"] for row in bins])
        top.errorbar(bin_confidence, bin_radius, yerr=bin_sem,
                     color="black", marker="o", markersize=5,
                     linewidth=2, capsize=3, label="bin mean ± SEM")
        pearson_value = pearson(radius, confidence)
        spearman_value = spearman(radius, confidence)
        top.set_title("{}: Pearson={}, Spearman={}".format(
            MODALITY_NAMES[name], _format_optional(pearson_value),
            _format_optional(spearman_value)))
        top.set_xlabel("Final SDT confidence")
        top.set_ylabel("Poincaré ball radius r = ||z||")
        top.set_xlim(0.0, 1.02)
        top.set_ylim(0.0, 1.0)
        top.grid(alpha=0.2)
        top.legend(loc="best")

        bottom = axes[1, column]
        correct_radius = radius[correctness]
        incorrect_radius = radius[~correctness]
        bottom.hist(correct_radius, bins=35, density=True, alpha=0.50,
                    color="tab:green", label="correct")
        bottom.hist(incorrect_radius, bins=35, density=True, alpha=0.50,
                    color="tab:red", label="incorrect")
        correct_mean = float(correct_radius.mean())
        incorrect_mean = float(incorrect_radius.mean())
        bottom.axvline(correct_mean, color="tab:green", linestyle="--")
        bottom.axvline(incorrect_mean, color="tab:red", linestyle="--")
        bottom.set_title(
            "mean correct={:.4f}, incorrect={:.4f}, Δ={:+.4f}".format(
                correct_mean, incorrect_mean,
                correct_mean - incorrect_mean))
        bottom.set_xlabel("Poincaré ball radius r = ||z||")
        bottom.set_xlim(0.0, 1.0)
        bottom.set_ylabel("Density")
        bottom.grid(alpha=0.2)
        bottom.legend(loc="best")
        radius_stats = summary_stats(radius)
        bottom.text(
            0.02, 0.97,
            "N={}\nmean±std={:.4f}±{:.4f}\nq05–q95={:.4f}–{:.4f}".format(
                radius_stats["count"], radius_stats["mean"],
                radius_stats["std"], radius_stats["q05"],
                radius_stats["q95"]),
            transform=bottom.transAxes, va="top", fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.78})

    figure.suptitle(
        "Poincaré ball radius (0 ≤ r < 1) versus final SDT confidence",
        fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(
        output_dir / "poincare_radius_confidence.png", dpi=200)
    plt.close(figure)


def flattened_report_row(report):
    row = {
        "id": report["experiment_id"], "dataset": report["dataset"],
        "seed": report["seed"], "epoch": report["epoch"],
        "geometry": report["geometry"], "dimension": report["dimension"],
        "radius_mode": report["radius_mode"],
        "fixed_radius": report["fixed_radius"],
        "zero_residual": report["zero_residual"],
        "weighted_f1": report["classification"]["weighted_f1"],
        "macro_f1": report["classification"]["macro_f1"],
        "accuracy": report["classification"]["accuracy"],
    }
    for name in MODALITIES:
        item = report["modalities"][name]
        radial = item["radial"]
        if radial["hyperbolic_radius"] is not None:
            radial_measure = radial["euclidean_norm"]
            radial_correct = radial["euclidean_norm_correct"]
            radial_incorrect = radial["euclidean_norm_incorrect"]
            radial_confidence = radial["norm_correlations"]["confidence"]
        else:
            radial_measure = radial["euclidean_norm"]
            radial_correct = radial["euclidean_norm_correct"]
            radial_incorrect = radial["euclidean_norm_incorrect"]
            radial_confidence = radial["norm_correlations"]["confidence"]
        row.update({
            name + "_wheel_accuracy": item["prototype"]["wheel_accuracy"],
            name + "_normalized_margin": item["prototype"][
                "normalized_margin"]["mean"],
            name + "_negative_margin_rate": item["prototype"][
                "negative_margin_rate"],
            name + "_semantic_cpcc": item["semantic_distance"][
                "class_pair_balanced_pearson_cpcc"],
            name + "_semantic_spearman": item["semantic_distance"][
                "class_pair_balanced_spearman"],
            name + "_ordering_accuracy": item["semantic_distance"][
                "ordering_accuracy_same_adjacent_far"],
            name + "_off_plane_ratio": item["off_plane"]["ratio"]["mean"],
            name + "_silhouette": item["cluster"]["silhouette"]["mean"],
            name + "_mean_rho": (
                radial["hyperbolic_radius"]["mean"]
                if radial["hyperbolic_radius"] is not None else None),
            name + "_radial_mean": radial_measure["mean"],
            name + "_radial_correct_mean": radial_correct["mean"],
            name + "_radial_incorrect_mean": radial_incorrect["mean"],
            name + "_radial_confidence_pearson": radial_confidence["pearson"],
            name + "_radial_confidence_spearman": radial_confidence["spearman"],
            name + "_boundary_gt_095": radial[
                "boundary_fraction_norm_gt_0_95"],
        })
    return row


def main(argv=None):
    args = build_parser().parse_args(argv)
    if (args.batch_size < 1 or args.max_batches < 0
            or args.max_utterances < 2 or args.pair_cap_per_class_pair < 0
            or args.distance_chunk_size < 1 or min(args.knn) < 1):
        raise ValueError("invalid positive/cap arguments")
    device = resolve_device(args.device, args.gpu_id)
    path = Path(args.path).expanduser().resolve()
    if path.is_file():
        checkpoints = [path]
    elif path.is_dir():
        checkpoints = sorted(path.rglob("best_checkpoint.pt"))
    else:
        raise FileNotFoundError(path)
    if not checkpoints:
        raise ValueError("no best_checkpoint.pt found under {}".format(path))
    if len(checkpoints) > 1 and args.output_dir:
        raise ValueError("--output-dir is only valid for one checkpoint")
    reports = []
    for checkpoint_path in checkpoints:
        try:
            reports.append(analyze_checkpoint(checkpoint_path, args, device))
        except ValueError as error:
            if path.is_file():
                raise
            print("Skipping {}: {}".format(checkpoint_path, error), flush=True)
    if not reports:
        raise ValueError("no wheel-enabled checkpoints were analysed")
    if path.is_dir():
        write_csv(path / ("geometry_diagnostics_" + args.split + ".csv"),
                  [flattened_report_row(report) for report in reports])
        print("Wrote combined diagnostics for {} checkpoints".format(
            len(reports)), flush=True)


if __name__ == "__main__":
    main()
