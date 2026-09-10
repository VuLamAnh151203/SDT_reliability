"""Load OOF reliability targets and create conservative pruning masks."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


MODALITIES = ("t", "a", "v")


class OOFReliabilityTable:
    def __init__(self, csv_path, modality_prune_quantile=0.0,
                 sample_prune_quantile=0.0, disagreement_weight=0.0,
                 sample_prune_any=False):
        for name, value in (("modality_prune_quantile", modality_prune_quantile),
                            ("sample_prune_quantile", sample_prune_quantile)):
            if not 0.0 <= value < 1.0:
                raise ValueError("{} must be in [0, 1)".format(name))
        if disagreement_weight < 0 or not np.isfinite(disagreement_weight):
            raise ValueError("disagreement_weight must be finite and nonnegative")
        self.rows = {}
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.sha256 = hashlib.sha256(self.csv_path.read_bytes()).hexdigest()
        metadata_path = self.csv_path.with_suffix(".json")
        self.metadata = (json.loads(metadata_path.read_text(encoding="utf-8"))
                         if metadata_path.is_file() else None)
        with self.csv_path.open(newline="", encoding="utf-8") as stream:
            for raw in csv.DictReader(stream):
                key = (raw["dialogue_id"], int(raw["utterance_index"]))
                if key in self.rows:
                    raise ValueError("duplicate OOF target: {}".format(key))
                reliability = np.asarray(
                    [float(raw[m + "_reliability"]) for m in MODALITIES],
                    dtype=np.float32)
                if (not np.isfinite(reliability).all() or
                        (reliability < 0).any() or reliability.sum() <= 0):
                    raise ValueError("invalid reliability target: {}".format(key))
                reliability /= reliability.sum()
                mean_ce = float(raw["mean_ce"])
                jsd = float(raw["jsd"])
                self.rows[key] = {
                    "reliability": reliability,
                    "label": int(raw["label"]),
                    "mean_ce": mean_ce,
                    "jsd": jsd,
                    "all_modalities_wrong": bool(int(raw["all_modalities_wrong"])),
                }
        if not self.rows:
            raise ValueError("OOF reliability target file is empty")

        reliabilities = np.stack([row["reliability"] for row in self.rows.values()])
        self.modality_thresholds = (
            np.quantile(reliabilities, modality_prune_quantile, axis=0)
            if modality_prune_quantile > 0 else np.full(3, -np.inf))
        labels = sorted({row["label"] for row in self.rows.values()})
        self.sample_thresholds = {}
        for label in labels:
            noise_scores = np.asarray([
                row["mean_ce"] + disagreement_weight * row["jsd"]
                for row in self.rows.values() if row["label"] == label
            ])
            self.sample_thresholds[label] = (
                float(np.quantile(noise_scores, 1.0 - sample_prune_quantile))
                if sample_prune_quantile > 0 else float("inf"))
        self.disagreement_weight = disagreement_weight
        self.sample_prune_any = sample_prune_any

    def validate_dialogues(self, dataset, dialogue_ids):
        if self.metadata is not None:
            if self.metadata.get("test_dialogues_used") is not False:
                raise ValueError("OOF metadata does not certify test exclusion")
            if self.metadata.get("model_config", {}).get("dataset") != dataset.dataset:
                raise ValueError("OOF metadata dataset does not match training dataset")
            if self.metadata.get("train_dialogues") != list(dataset.trainVid):
                raise ValueError("OOF metadata train dialogue order does not match dataset")
            if self.metadata.get("written_utterances") != len(self.rows):
                raise ValueError("OOF metadata row count does not match CSV")
        missing = []
        mismatched_labels = []
        for dialogue_id in dialogue_ids:
            for utterance_index in range(len(dataset.videoLabels[dialogue_id])):
                key = (dialogue_id, utterance_index)
                if key not in self.rows:
                    missing.append(key)
                    if len(missing) == 5:
                        break
                elif self.rows[key]["label"] != int(
                        dataset.videoLabels[dialogue_id][utterance_index]):
                    mismatched_labels.append(key)
                    if len(mismatched_labels) == 5:
                        break
            if len(missing) == 5:
                break
        if missing:
            raise ValueError("OOF targets are missing training utterances: {}".format(missing))
        if mismatched_labels:
            raise ValueError("OOF target labels differ from dataset: {}".format(
                mismatched_labels))

    def batch(self, dialogue_ids, lengths, max_length, device):
        batch_size = len(dialogue_ids)
        targets = torch.full((batch_size, max_length, 3), 1.0 / 3.0,
                             dtype=torch.float32, device=device)
        sample_keep = torch.zeros(batch_size, max_length, dtype=torch.float32,
                                  device=device)
        modality_keep = torch.zeros(batch_size, max_length, 3,
                                    dtype=torch.float32, device=device)
        for batch_index, (dialogue_id, length) in enumerate(zip(dialogue_ids, lengths)):
            for utterance_index in range(length):
                key = (dialogue_id, utterance_index)
                if key not in self.rows:
                    raise ValueError("missing OOF reliability target: {}".format(key))
                row = self.rows[key]
                reliability = row["reliability"]
                targets[batch_index, utterance_index] = torch.as_tensor(
                    reliability, device=device)
                keep = reliability >= self.modality_thresholds
                # Never prune every modality; always retain the OOF-best one.
                keep[int(reliability.argmax())] = True
                modality_keep[batch_index, utterance_index] = torch.as_tensor(
                    keep, dtype=torch.float32, device=device)
                noise_score = (row["mean_ce"] +
                               self.disagreement_weight * row["jsd"])
                suspected = noise_score > self.sample_thresholds[row["label"]]
                if not self.sample_prune_any:
                    suspected = suspected and row["all_modalities_wrong"]
                sample_keep[batch_index, utterance_index] = 0.0 if suspected else 1.0
        return targets, sample_keep, modality_keep

    def summary(self):
        reliabilities = np.stack([row["reliability"] for row in self.rows.values()])
        return {
            "source_path": str(self.csv_path),
            "sha256": self.sha256,
            "metadata_verified": self.metadata is not None,
            "utterances": len(self.rows),
            "mean_reliability": {
                name: float(reliabilities[:, index].mean())
                for index, name in enumerate(MODALITIES)
            },
            "modality_thresholds": {
                name: (float(self.modality_thresholds[index])
                       if np.isfinite(self.modality_thresholds[index]) else None)
                for index, name in enumerate(MODALITIES)
            },
            "sample_noise_thresholds_by_class": {
                str(label): (threshold if np.isfinite(threshold) else None)
                for label, threshold in self.sample_thresholds.items()
            },
            "disagreement_weight": self.disagreement_weight,
            "sample_prune_any": self.sample_prune_any,
        }
