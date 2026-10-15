"""Linear-probe pure and enhanced SDT representations from checkpoints."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataloader import DialogueDataset, make_loaders
from model import MODALITIES, Transformer_Based_Model
from tical import circular_class_distance_matrix, emotion_wheel_angles


def torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def checkpoints(target):
    target = target.resolve()
    return [target] if target.is_file() else sorted(target.rglob("best_checkpoint.pt"))


@torch.no_grad()
def extract(model, loader, device):
    store = {source: {name: [] for name in MODALITIES}
             for source in ("pure", "enhanced")}
    labels_all = []
    model.eval()
    for batch in loader:
        text, visual, audio, speakers, mask, labels = [x.to(device) for x in batch[:6]]
        valid = mask.bool()
        lengths = valid.sum(dim=1).tolist()
        output = model(text, visual, audio, mask, speakers.transpose(0, 1), lengths)
        for source in store:
            tensor = output[source]
            if tensor is None:
                raise RuntimeError("checkpoint did not expose {} features".format(source))
            for index, name in enumerate(MODALITIES):
                store[source][name].append(tensor[..., index, :][valid].cpu().numpy())
        labels_all.append(labels[valid].cpu().numpy())
    return ({source: {name: np.concatenate(values) for name, values in modalities.items()}
             for source, modalities in store.items()}, np.concatenate(labels_all))


def centroid_spearman(features, labels, dataset, n_classes):
    centroids = np.stack([features[labels == index].mean(axis=0)
                          for index in range(n_classes)])
    norms = np.linalg.norm(centroids, axis=1, keepdims=True).clip(1e-12)
    cosine = 1.0 - (centroids / norms) @ (centroids / norms).T
    wheel = circular_class_distance_matrix(
        emotion_wheel_angles(dataset, n_classes)).numpy()
    row, column = np.triu_indices(n_classes, 1)
    return float(spearmanr(cosine[row, column], wheel[row, column]).statistic)


def analyze(checkpoint_path, device, feature_path=None, batch_size=16):
    checkpoint = torch_load(checkpoint_path, device)
    config = dict(checkpoint["model_config"])
    dataset_name = config["dataset"]
    saved_feature = checkpoint.get("feature_path")
    dataset = DialogueDataset(dataset_name, feature_path or saved_feature)
    # Baseline SDT checkpoints do not request pure features during forward.
    # Switching the source only exposes H_TT/H_AA/H_VV; it does not alter them.
    if not config.get("use_tical", False):
        config["fusion_feature_source"] = "pure"
    model = Transformer_Based_Model(**config).to(device)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = [key for key in incompatible.missing_keys
               if not key.endswith("wheel_label_permutation")]
    if unexpected or missing:
        raise RuntimeError("checkpoint mismatch: missing={} unexpected={}".format(
            missing, unexpected))
    split_ids = checkpoint["split_ids"]
    loaders = make_loaders(dataset, split_ids, batch_size, seed=0,
                           num_workers=0, pin_memory=device.type == "cuda")
    train_x, train_y = extract(model, loaders["train"], device)
    test_x, test_y = extract(model, loaders["test"], device)
    rows = []
    for source in ("pure", "enhanced"):
        for name in MODALITIES:
            classifier = make_pipeline(
                StandardScaler(), LogisticRegression(
                    max_iter=2000, solver="lbfgs", random_state=0))
            classifier.fit(train_x[source][name], train_y)
            prediction = classifier.predict(test_x[source][name])
            rows.append({
                "checkpoint": str(checkpoint_path.resolve()),
                "source": source, "modality": name,
                "accuracy": 100.0 * accuracy_score(test_y, prediction),
                "weighted_f1": 100.0 * f1_score(
                    test_y, prediction, average="weighted", zero_division=0),
                "macro_f1": 100.0 * f1_score(
                    test_y, prediction, average="macro", zero_division=0),
                "centroid_wheel_spearman": centroid_spearman(
                    test_x[source][name], test_y, dataset_name,
                    dataset.n_classes),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="checkpoint or result directory")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--feature-path")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda", args.gpu_id) if args.device == "cuda" else torch.device("cpu")
    paths = checkpoints(args.target)
    if not paths:
        raise SystemExit("No best_checkpoint.pt found")
    rows = []
    for path in paths:
        print("Probing {}".format(path), flush=True)
        rows.extend(analyze(path, device, args.feature_path, args.batch_size))
    if args.output is not None:
        output = args.output.resolve()
    elif args.target.is_dir():
        output = args.target.resolve() / "representation_probes.csv"
    else:
        output = args.target.resolve().with_name("representation_probes.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print("Wrote {}".format(output))


if __name__ == "__main__":
    main()

