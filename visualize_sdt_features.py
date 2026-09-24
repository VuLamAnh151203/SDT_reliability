"""Visualize original SDT representations and modality-gate weights."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataloader import DialogueDataset, make_loaders
from model import MODALITIES, Transformer_Based_Model


CLASS_NAMES = {
    "IEMOCAP": ("happy", "sad", "neutral", "angry", "excited", "frustrated"),
    "MELD": ("neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"),
}
REPRESENTATIONS = (
    ("pure_t", "Pure Text $H_{TT}$"),
    ("pure_a", "Pure Audio $H_{AA}$"),
    ("pure_v", "Pure Visual $H_{VV}$"),
    ("enhanced_t", "Enhanced Text $H'_T$"),
    ("enhanced_a", "Enhanced Audio $H'_A$"),
    ("enhanced_v", "Enhanced Visual $H'_V$"),
    ("fused", "SDT fused feature"),
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="path to an original-SDT best_checkpoint.pt")
    parser.add_argument("--feature-path",
                        help="override feature pickle saved in the checkpoint")
    parser.add_argument("--split", choices=("train", "valid", "test"),
                        default="test")
    parser.add_argument("--output-prefix",
                        help="output prefix; defaults beside the checkpoint")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-points-per-class", type=int, default=300,
                        help="plot cap per class; PCA and CSV still use all collected points")
    parser.add_argument("--max-batches", type=int, default=0,
                        help="limit batches for a smoke visualization; 0 uses the full split")
    parser.add_argument("--poincare-target-radius", type=float, default=0.85,
                        help="map the configured PCA-norm quantile to this ball radius")
    parser.add_argument("--poincare-scale-quantile", type=float, default=0.95,
                        help="PCA tangent-norm quantile used to set the exp-map scale")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
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


def collect_features(model, loader, device, max_batches=0):
    features = {name: [] for name, _ in REPRESENTATIONS}
    metadata = []
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            text, visual, audio, speakers, mask, labels = [
                item.to(device) for item in batch[:6]
            ]
            valid = mask.bool()
            lengths = valid.sum(dim=1).tolist()
            enhanced, pure = model.encode_modalities(
                text, visual, audio, mask, speakers.transpose(0, 1),
                lengths, return_pure=True)
            stacked = torch.stack(enhanced, dim=-2)
            gate_weights = torch.softmax(model.last_gate.fc(stacked), dim=-2)
            fused = (gate_weights * stacked).sum(dim=-2)
            prediction = model.all_output_layer(fused).argmax(dim=-1)
            tensors = {
                "pure_t": pure[0], "pure_a": pure[1], "pure_v": pure[2],
                "enhanced_t": enhanced[0], "enhanced_a": enhanced[1],
                "enhanced_v": enhanced[2], "fused": fused,
            }
            for name in features:
                features[name].append(tensors[name][valid].float().cpu())
            scalar_gate = gate_weights.mean(dim=-1)
            for item_index, (dialogue_id, length) in enumerate(zip(batch[6], lengths)):
                for utterance_index in range(length):
                    gates = scalar_gate[item_index, utterance_index].cpu().tolist()
                    metadata.append({
                        "dialogue_id": dialogue_id,
                        "utterance_index": utterance_index,
                        "label": int(labels[item_index, utterance_index]),
                        "prediction": int(prediction[item_index, utterance_index]),
                        "gate_t": gates[0],
                        "gate_a": gates[1],
                        "gate_v": gates[2],
                    })
    if not metadata:
        raise ValueError("selected split contains no utterances")
    return {name: torch.cat(chunks).numpy() for name, chunks in features.items()}, metadata


def fit_pca(features, metadata, dataset_name):
    rows = [dict(row, emotion=CLASS_NAMES[dataset_name][row["label"]])
            for row in metadata]
    explained = {}
    for name, _ in REPRESENTATIONS:
        estimator = PCA(n_components=2)
        transformed = estimator.fit_transform(features[name])
        explained[name] = float(estimator.explained_variance_ratio_.sum())
        for row, coordinate in zip(rows, transformed):
            row[name + "_x"] = float(coordinate[0])
            row[name + "_y"] = float(coordinate[1])
    return rows, explained


def add_poincare_coordinates(rows, target_radius, scale_quantile):
    """Map each panel's two PCA coordinates through Exp_0 into the unit disk."""
    scales = {}
    target_tangent_norm = np.arctanh(target_radius)
    for representation, _ in REPRESENTATIONS:
        tangent = np.asarray([
            [row[representation + "_x"], row[representation + "_y"]]
            for row in rows
        ], dtype=np.float64)
        raw_norm = np.linalg.norm(tangent, axis=1)
        reference = float(np.quantile(raw_norm, scale_quantile))
        scale = target_tangent_norm / max(reference, 1e-12)
        tangent = tangent * scale
        tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
        ball = (np.tanh(tangent_norm)
                * tangent / np.maximum(tangent_norm, 1e-12))
        ball[tangent_norm[:, 0] <= 1e-12] = 0.0
        scales[representation] = scale
        for row, coordinate in zip(rows, ball):
            row[representation + "_poincare_x"] = float(coordinate[0])
            row[representation + "_poincare_y"] = float(coordinate[1])
            row[representation + "_poincare_norm"] = float(
                np.linalg.norm(coordinate))
    return scales


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sampled_indices(rows, n_classes, maximum, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for class_index in range(n_classes):
        indices = np.asarray([
            index for index, row in enumerate(rows)
            if row["label"] == class_index
        ])
        if maximum > 0 and len(indices) > maximum:
            indices = rng.choice(indices, maximum, replace=False)
        selected.extend(indices.tolist())
    return np.asarray(sorted(selected))


def render_feature_pca(path, rows, explained, dataset_name, maximum, seed):
    names = CLASS_NAMES[dataset_name]
    colors = plt.get_cmap("tab10")(np.arange(len(names)))
    indices = sampled_indices(rows, len(names), maximum, seed)
    figure, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    for axis, (representation, title) in zip(axes, REPRESENTATIONS):
        for class_index, emotion in enumerate(names):
            group = [index for index in indices
                     if rows[index]["label"] == class_index]
            axis.scatter(
                [rows[index][representation + "_x"] for index in group],
                [rows[index][representation + "_y"] for index in group],
                s=12, alpha=0.34, color=colors[class_index],
                edgecolors="none", label=emotion)
            if group:
                center_x = np.mean([
                    rows[index][representation + "_x"] for index in group])
                center_y = np.mean([
                    rows[index][representation + "_y"] for index in group])
                axis.scatter(center_x, center_y, marker="X", s=105,
                             color=colors[class_index], edgecolors="black",
                             linewidths=0.7, zorder=4)
        axis.set_title("{}\nexplained variance={:.1%}".format(
            title, explained[representation]))
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.grid(alpha=0.15)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(names),
                  bbox_to_anchor=(0.5, 0.01), frameon=False)
    figure.suptitle(
        "{} original SDT representations (independent PCA per panel)".format(
            dataset_name), fontsize=16, y=0.99)
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.96))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def render_feature_poincare(path, rows, dataset_name, maximum, seed,
                            target_radius, scale_quantile):
    names = CLASS_NAMES[dataset_name]
    colors = plt.get_cmap("tab10")(np.arange(len(names)))
    indices = sampled_indices(rows, len(names), maximum, seed)
    circle_angle = np.linspace(0.0, 2.0 * math.pi, 512)
    figure, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    for axis, (representation, title) in zip(axes, REPRESENTATIONS):
        axis.plot(np.cos(circle_angle), np.sin(circle_angle),
                  color="black", linewidth=1.2)
        axis.plot(target_radius * np.cos(circle_angle),
                  target_radius * np.sin(circle_angle),
                  color="gray", linewidth=0.8, linestyle="--")
        for class_index, emotion in enumerate(names):
            group = [index for index in indices
                     if rows[index]["label"] == class_index]
            x_key = representation + "_poincare_x"
            y_key = representation + "_poincare_y"
            axis.scatter(
                [rows[index][x_key] for index in group],
                [rows[index][y_key] for index in group],
                s=12, alpha=0.34, color=colors[class_index],
                edgecolors="none", label=emotion)
            if group:
                center_x = np.mean([rows[index][x_key] for index in group])
                center_y = np.mean([rows[index][y_key] for index in group])
                axis.scatter(center_x, center_y, marker="X", s=105,
                             color=colors[class_index], edgecolors="black",
                             linewidths=0.7, zorder=4)
        mean_radius = np.mean([
            row[representation + "_poincare_norm"] for row in rows])
        axis.set_title("{}\nmean ball radius={:.3f}".format(
            title, mean_radius))
        axis.set_xlim(-1.06, 1.06)
        axis.set_ylim(-1.06, 1.06)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Poincaré dimension 1")
        axis.set_ylabel("Poincaré dimension 2")
        axis.grid(alpha=0.15)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(names),
                  bbox_to_anchor=(0.5, 0.01), frameon=False)
    figure.suptitle(
        ("{} original SDT: post-hoc PCA tangent projection → "
         "Poincaré Exp₀\n"
         "dashed radius={:.2f} at PCA norm quantile={:.0%}").format(
            dataset_name, target_radius, scale_quantile),
        fontsize=15, y=0.99)
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.94))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def render_gate_triangle(path, rows, dataset_name, maximum, seed):
    names = CLASS_NAMES[dataset_name]
    colors = plt.get_cmap("tab10")(np.arange(len(names)))
    indices = sampled_indices(rows, len(names), maximum, seed)
    sqrt_three = math.sqrt(3.0)
    vertices = {
        "Text": np.asarray([0.0, 1.0]),
        "Audio": np.asarray([-sqrt_three / 2.0, -0.5]),
        "Visual": np.asarray([sqrt_three / 2.0, -0.5]),
    }
    figure, axis = plt.subplots(figsize=(9, 8))
    outline = np.vstack((list(vertices.values()), vertices["Text"]))
    axis.plot(outline[:, 0], outline[:, 1], color="black", linewidth=1.2)
    for label, coordinate in vertices.items():
        axis.text(coordinate[0] * 1.09, coordinate[1] * 1.09, label,
                  ha="center", va="center", fontsize=12, fontweight="bold")
    for class_index, emotion in enumerate(names):
        group = [index for index in indices
                 if rows[index]["label"] == class_index]
        coordinates = []
        for index in group:
            row = rows[index]
            coordinates.append(
                row["gate_t"] * vertices["Text"]
                + row["gate_a"] * vertices["Audio"]
                + row["gate_v"] * vertices["Visual"])
        if coordinates:
            coordinates = np.asarray(coordinates)
            axis.scatter(coordinates[:, 0], coordinates[:, 1], s=14,
                         alpha=0.36, color=colors[class_index],
                         edgecolors="none", label=emotion)
    axis.set_title(
        "{} original SDT modality gate\n"
        "each point is the feature-wise mean gate of one utterance".format(
            dataset_name))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-1.03, 1.03)
    axis.set_ylim(-0.63, 1.13)
    axis.axis("off")
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.10),
                ncol=3, frameon=False)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if (args.batch_size < 1 or args.max_points_per_class < 0
            or args.max_batches < 0):
        raise ValueError("batch size must be positive and limits nonnegative")
    if not 0.0 < args.poincare_target_radius < 1.0:
        raise ValueError("--poincare-target-radius must be in (0,1)")
    if not 0.0 < args.poincare_scale_quantile <= 1.0:
        raise ValueError("--poincare-scale-quantile must be in (0,1]")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    device = resolve_device(args.device, args.gpu_id)
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint["model_config"]
    if config.get("fusion_variant") != "sdt" or config.get("use_tical", False):
        raise ValueError(
            "checkpoint must be original SDT: fusion_variant=sdt and use_tical=false")
    dataset_name = config["dataset"]
    feature_path = args.feature_path or checkpoint.get("feature_path")
    dataset = DialogueDataset(dataset_name, feature_path)
    split_ids = checkpoint["split_ids"]
    if args.split == "valid" and not split_ids.get("valid"):
        raise ValueError("checkpoint has no validation split")
    loaders = make_loaders(
        dataset, split_ids, args.batch_size, args.seed,
        num_workers=0, pin_memory=device.type == "cuda")
    model = Transformer_Based_Model(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    features, metadata = collect_features(
        model, loaders[args.split], device, args.max_batches)
    rows, explained = fit_pca(features, metadata, dataset_name)
    add_poincare_coordinates(
        rows, args.poincare_target_radius, args.poincare_scale_quantile)

    if args.output_prefix:
        prefix = Path(args.output_prefix).expanduser().resolve()
    else:
        prefix = checkpoint_path.with_name("sdt_{}_features".format(args.split))
    prefix.parent.mkdir(parents=True, exist_ok=True)
    feature_path_out = prefix.with_name(prefix.name + "_pca.png")
    poincare_path_out = prefix.with_name(prefix.name + "_poincare.png")
    gate_path_out = prefix.with_name(prefix.name + "_gate.png")
    csv_path = prefix.with_suffix(".csv")
    write_csv(csv_path, rows)
    render_feature_pca(
        feature_path_out, rows, explained, dataset_name,
        args.max_points_per_class, args.seed)
    render_feature_poincare(
        poincare_path_out, rows, dataset_name,
        args.max_points_per_class, args.seed,
        args.poincare_target_radius, args.poincare_scale_quantile)
    render_gate_triangle(
        gate_path_out, rows, dataset_name,
        args.max_points_per_class, args.seed)
    print("Saved feature PCA: {}".format(feature_path_out))
    print("Saved Poincare projection: {}".format(poincare_path_out))
    print("Saved gate triangle: {}".format(gate_path_out))
    print("Saved coordinates: {}".format(csv_path))


if __name__ == "__main__":
    main()
