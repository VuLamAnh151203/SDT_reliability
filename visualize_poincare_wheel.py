"""Visualize learned modality embeddings and fixed emotion-wheel prototypes."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataloader import DialogueDataset, make_loaders
from model import MODALITIES, Transformer_Based_Model


CLASS_NAMES = {
    "IEMOCAP": ("happy", "sad", "neutral", "angry", "excited", "frustrated"),
    "MELD": ("neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"),
}
MODALITY_TITLES = {"t": "Text", "a": "Audio", "v": "Visual"}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="path to a wheel-enabled best_checkpoint.pt")
    parser.add_argument("--feature-path",
                        help="override feature pickle saved in the checkpoint")
    parser.add_argument("--split", choices=("train", "valid", "test"),
                        default="test")
    parser.add_argument("--output", help="PNG path; defaults beside checkpoint")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-points-per-class", type=int, default=300,
                        help="plot cap per class and modality; CSV always keeps all points")
    parser.add_argument("--max-batches", type=int, default=0,
                        help="limit batches for a smoke visualization; 0 uses the full split")
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
    except TypeError:  # PyTorch versions before weights_only was added.
        return torch.load(path, map_location=device)


def collect_embeddings(model, loader, device, dataset_name, max_batches=0):
    rows = []
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
            output = model(
                text, visual, audio, mask, speakers.transpose(0, 1), lengths)
            tical = output.get("tical")
            if tical is None or not tical.get("wheel_enabled", False):
                raise ValueError("checkpoint does not contain an enabled emotion wheel")
            final_prediction = output["logits"].argmax(dim=-1)
            offset = 0
            for batch_index, (dialogue_id, length) in enumerate(zip(batch[6], lengths)):
                for utterance_index in range(length):
                    label = int(labels[batch_index, utterance_index])
                    for name in MODALITIES:
                        point = tical["projected"][name][batch_index, utterance_index]
                        coordinates = point.detach().float().cpu().tolist()
                        wheel_prediction = int(
                            tical["wheel_pseudo_labels"][name][offset])
                        row = {
                            "dialogue_id": dialogue_id,
                            "utterance_index": utterance_index,
                            "modality": name,
                            "label": label,
                            "emotion": CLASS_NAMES[dataset_name][label],
                            "final_prediction": int(
                                final_prediction[batch_index, utterance_index]),
                            "wheel_prediction": wheel_prediction,
                            "wheel_emotion": CLASS_NAMES[dataset_name][wheel_prediction],
                            "x": coordinates[0],
                            "y": coordinates[1],
                            "full_norm": float(point.norm().item()),
                            "off_plane_norm": float(point[2:].norm().item()),
                        }
                        row.update({
                            "z_{}".format(index): value
                            for index, value in enumerate(coordinates)
                        })
                        rows.append(row)
                    offset += 1
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sampled_rows(rows, modality, n_classes, maximum, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for class_index in range(n_classes):
        group = [row for row in rows
                 if row["modality"] == modality and row["label"] == class_index]
        if maximum > 0 and len(group) > maximum:
            indices = np.sort(rng.choice(len(group), maximum, replace=False))
            group = [group[index] for index in indices]
        selected.extend(group)
    return selected


def render(path, rows, prototypes, dataset_name, maximum, seed):
    names = CLASS_NAMES[dataset_name]
    colors = plt.get_cmap("tab10")(np.arange(len(names)))
    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    angle = np.linspace(0.0, 2.0 * math.pi, 512)
    prototype_xy = prototypes[:, :2].detach().float().cpu().numpy()
    prototype_radius = np.linalg.norm(prototype_xy, axis=1).mean()

    for modal_index, (axis, modality) in enumerate(zip(axes, MODALITIES)):
        axis.plot(np.cos(angle), np.sin(angle), color="black", linewidth=1.3)
        axis.plot(prototype_radius * np.cos(angle),
                  prototype_radius * np.sin(angle), color="gray",
                  linewidth=0.8, linestyle="--")
        points = sampled_rows(
            rows, modality, len(names), maximum, seed + modal_index)
        for class_index, name in enumerate(names):
            group = [row for row in points if row["label"] == class_index]
            if group:
                axis.scatter(
                    [row["x"] for row in group],
                    [row["y"] for row in group],
                    s=13, alpha=0.34, color=colors[class_index],
                    edgecolors="none", label=name)
        axis.scatter(prototype_xy[:, 0], prototype_xy[:, 1], marker="*",
                     s=260, color=colors, edgecolors="black", linewidths=0.9,
                     zorder=5)
        for class_index, name in enumerate(names):
            x, y = prototype_xy[class_index]
            axis.annotate(name, (x, y), xytext=(1.11 * x, 1.11 * y),
                          ha="center", va="center", fontsize=9,
                          fontweight="bold", color=colors[class_index])
        modality_rows = [row for row in rows if row["modality"] == modality]
        off_plane = np.mean([row["off_plane_norm"] for row in modality_rows])
        full_norm = np.mean([row["full_norm"] for row in modality_rows])
        wheel_accuracy = np.mean([
            row["label"] == row["wheel_prediction"] for row in modality_rows
        ])
        axis.set_title(
            "{} | wheel acc={:.1%}\nmean norm={:.3f}, off-plane={:.3f}".format(
                MODALITY_TITLES[modality], wheel_accuracy,
                full_norm, off_plane))
        axis.set_xlim(-1.08, 1.08)
        axis.set_ylim(-1.08, 1.08)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Poincaré dimension 1")
        axis.set_ylabel("Poincaré dimension 2")
        axis.grid(alpha=0.15)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(names),
                  bbox_to_anchor=(0.5, 0.01), frameon=False)
    figure.suptitle(
        "{} emotion-wheel plane (color = ground-truth emotion)".format(
            dataset_name), fontsize=15, y=0.98)
    figure.tight_layout(rect=(0.0, 0.10, 1.0, 0.94))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if (args.batch_size < 1 or args.max_points_per_class < 0
            or args.max_batches < 0):
        raise ValueError("batch size must be positive and limits nonnegative")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    device = resolve_device(args.device, args.gpu_id)
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint["model_config"]
    if not config.get("use_tical") or not config.get("use_emotion_wheel"):
        raise ValueError("checkpoint must have use_tical=true and use_emotion_wheel=true")
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
    model.set_tical_epoch(checkpoint.get("epoch", 0))
    rows = collect_embeddings(
        model, loaders[args.split], device, dataset_name, args.max_batches)
    if not rows:
        raise ValueError("selected split contains no utterances")

    output_path = (Path(args.output).expanduser().resolve() if args.output else
                   checkpoint_path.with_name(
                       "poincare_wheel_{}.png".format(args.split)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.with_suffix(".csv")
    write_csv(csv_path, rows)
    render(output_path, rows, model.tical.wheel_prototypes,
           dataset_name, args.max_points_per_class, args.seed)
    print("Saved figure: {}".format(output_path))
    print("Saved coordinates: {}".format(csv_path))


if __name__ == "__main__":
    main()
