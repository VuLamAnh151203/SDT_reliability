"""Measure SDT/Wheel gradient cosine on the Wheel-reachable pure encoders."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataloader import DialogueDataset, make_loaders
from model import Transformer_Based_Model
from train import build_parser, make_criterion


GROUP_PREFIXES = {
    "text": ("textf_input.", "t_t."),
    "audio": ("acouf_input.", "a_a."),
    "visual": ("visuf_input.", "v_v."),
}


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def gradient_vector(loss, parameters, retain_graph):
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return [gradient if gradient is not None else torch.zeros_like(parameter)
            for parameter, gradient in zip(parameters, gradients)]


def cosine(first, second):
    if not first or not second or len(first) != len(second):
        return float("nan"), 0.0, 0.0
    dot = sum((a * b).sum() for a, b in zip(first, second))
    norm_a = torch.sqrt(sum(a.square().sum() for a in first))
    norm_b = torch.sqrt(sum(b.square().sum() for b in second))
    if norm_a.item() == 0 or norm_b.item() == 0:
        return float("nan"), float(norm_a), float(norm_b)
    return float(dot / (norm_a * norm_b)), float(norm_a), float(norm_b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--feature-path")
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda", args.gpu_id) if args.device == "cuda" else torch.device("cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["model_config"]
    model = Transformer_Based_Model(**config).to(device)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    missing = [key for key in incompatible.missing_keys
               if not key.endswith("wheel_label_permutation")]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError("checkpoint state mismatch")
    model.eval(); model.set_tical_epoch(checkpoint.get("epoch", 0))

    saved_args = build_parser().parse_args([])
    for key, value in checkpoint["args"].items():
        setattr(saved_args, key, value)
    saved_args.dataset = config["dataset"]
    criterion = make_criterion(saved_args, device)
    dataset = DialogueDataset(config["dataset"], args.feature_path or checkpoint.get("feature_path"))
    loader = make_loaders(dataset, checkpoint["split_ids"],
                          checkpoint["args"].get("batch_size", 16), 0)["train"]
    groups = {group: [parameter for name, parameter in model.named_parameters()
                      if any(name.startswith(prefix) for prefix in prefixes)]
              for group, prefixes in GROUP_PREFIXES.items()}
    rows = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.batches:
            break
        text, visual, audio, speakers, mask, labels = [x.to(device) for x in batch[:6]]
        valid = mask.bool(); lengths = valid.sum(dim=1).tolist()
        output = model(text, visual, audio, mask, speakers.transpose(0, 1), lengths)
        _, parts, _ = criterion(output, labels, valid)
        wheel = parts["weighted_wheel_proto"] + parts["weighted_wheel_cpcc"]
        for group, parameters in groups.items():
            sdt_gradient = gradient_vector(parts["sdt"], parameters, True)
            wheel_gradient = gradient_vector(wheel, parameters, True)
            value, norm_sdt, norm_wheel = cosine(sdt_gradient, wheel_gradient)
            rows.append({"batch": batch_index, "group": group,
                         "cosine": value, "sdt_grad_norm": norm_sdt,
                         "wheel_grad_norm": norm_wheel})
    output = args.output or args.checkpoint.with_name("gradient_alignment.csv")
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {}
    for group in GROUP_PREFIXES:
        values = [row["cosine"] for row in rows
                  if row["group"] == group and math.isfinite(row["cosine"])]
        summary[group] = {
            "batches": len(values),
            "mean_cosine": sum(values) / len(values) if values else None,
            "positive_fraction": (sum(value > 0 for value in values) / len(values)
                                  if values else None),
        }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)); print("Wrote {}".format(output))


if __name__ == "__main__":
    main()

