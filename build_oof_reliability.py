"""Build leakage-free modality reliability targets from SDT OOF predictions."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold

from dataloader import BASE_DIR, DialogueDataset, make_loaders
from losses import SDTCOLDLoss
from model import MODALITIES, Transformer_Based_Model
from train import run_epoch, seed_everything


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--Dataset", "--dataset", dest="dataset",
                        choices=("IEMOCAP", "MELD"), default="IEMOCAP")
    result.add_argument("--feature-path")
    result.add_argument("--folds", type=int, default=5)
    result.add_argument("--epochs", type=int, default=50,
                        help="fixed training budget per OOF fold; no holdout selection")
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--hidden-dim", type=int, default=1024)
    result.add_argument("--n-head", type=int, default=8)
    result.add_argument("--dropout", type=float, default=0.5)
    result.add_argument("--lr", type=float, default=1e-4)
    result.add_argument("--l2", type=float, default=1e-5)
    result.add_argument("--temp", type=float, default=1.0)
    result.add_argument("--reliability-temperature", type=float, default=1.0)
    result.add_argument("--no-class-weight", action="store_true")
    result.add_argument("--seed", type=int, default=2024)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--gpu-id", type=int, default=0)
    result.add_argument("--num-workers", type=int, default=0)
    result.add_argument("--num-threads", type=int, default=0)
    result.add_argument("--grad-clip", type=float, default=0.0)
    result.add_argument("--max-batches", type=int, default=0,
                        help="debug only; produces an intentionally incomplete manifest")
    result.add_argument("--output", default=str(
        BASE_DIR / "reliability_targets" / "iemocap_oof_seed2024.csv"))
    result.add_argument("--overwrite", action="store_true")
    return result


def resolve_device(args):
    if args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda", args.gpu_id)


def class_weights(args, device):
    if args.dataset != "IEMOCAP" or args.no_class_weight:
        return None
    return torch.tensor([1 / p for p in (0.086747, 0.144406, 0.227883,
                                        0.160585, 0.127711, 0.252668)],
                        dtype=torch.float32, device=device)


def model_config(args, dataset):
    return {"dataset": args.dataset, "temp": args.temp, **dataset.feature_dims,
            "n_head": args.n_head, "n_classes": dataset.n_classes,
            "hidden_dim": args.hidden_dim, "n_speakers": dataset.n_speakers,
            "dropout": args.dropout, "fusion_variant": "sdt"}


def collect_fold(model, loader, device, fold, reliability_temperature):
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            text, visual, audio, speakers, mask, labels = [item.to(device) for item in batch[:6]]
            valid = mask.bool()
            lengths = valid.sum(dim=1).tolist()
            output = model(text, visual, audio, mask, speakers.transpose(0, 1), lengths)
            valid_labels = labels[valid]
            logits = torch.stack([
                output["student_logits"][name][valid] for name in MODALITIES
            ], dim=1)
            probabilities = torch.softmax(logits, dim=-1)
            errors = torch.stack([
                F.cross_entropy(logits[:, index], valid_labels, reduction="none")
                for index in range(len(MODALITIES))
            ], dim=-1)
            reliability = torch.softmax(-errors / reliability_temperature, dim=-1)
            mean_probability = probabilities.mean(dim=1)
            jsd = (probabilities * (
                probabilities.clamp_min(1e-12).log() -
                mean_probability.unsqueeze(1).clamp_min(1e-12).log()
            )).sum(dim=-1).mean(dim=-1)
            modal_predictions = probabilities.argmax(dim=-1)
            fused_prediction = output["logits"][valid].argmax(dim=-1)
            cursor = 0
            for batch_index, (dialogue_id, length) in enumerate(zip(batch[6], lengths)):
                for utterance_index in range(length):
                    label = int(valid_labels[cursor])
                    predictions = modal_predictions[cursor]
                    row = {
                        "dialogue_id": dialogue_id,
                        "utterance_index": utterance_index,
                        "label": label,
                        "fold": fold,
                        "fused_prediction": int(fused_prediction[cursor]),
                        "mean_ce": float(errors[cursor].mean()),
                        "jsd": float(jsd[cursor]),
                        "all_modalities_wrong": int(bool((predictions != label).all())),
                        "unanimous_wrong": int(bool(
                            (predictions != label).all() and
                            (predictions == predictions[0]).all())),
                    }
                    for modal_index, name in enumerate(MODALITIES):
                        row[name + "_prediction"] = int(predictions[modal_index])
                        row[name + "_ce"] = float(errors[cursor, modal_index])
                        row[name + "_label_probability"] = float(
                            probabilities[cursor, modal_index, label])
                        row[name + "_reliability"] = float(
                            reliability[cursor, modal_index])
                    rows.append(row)
                    cursor += 1
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.folds < 2 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("folds must be >=2; epochs and batch size must be positive")
    if args.reliability_temperature <= 0 or args.lr <= 0 or args.l2 < 0:
        raise ValueError("temperatures/lr must be positive and l2 nonnegative")
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    output_path = Path(args.output).expanduser().resolve()
    metadata_path = output_path.with_suffix(".json")
    if (output_path.exists() or metadata_path.exists()) and not args.overwrite:
        raise FileExistsError("output exists; choose another --output or pass --overwrite")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args)
    dataset = DialogueDataset(args.dataset, args.feature_path)
    if args.folds > len(dataset.trainVid):
        raise ValueError("more folds than training dialogues")
    config = model_config(args, dataset)
    splitter = KFold(args.folds, shuffle=True, random_state=args.seed)
    dialogue_array = np.asarray(dataset.trainVid, dtype=object)
    all_rows = []
    fold_metadata = []
    for fold_index, (train_indices, holdout_indices) in enumerate(
            splitter.split(dialogue_array), start=1):
        fold_seed = args.seed + fold_index - 1
        seed_everything(fold_seed)
        train_ids = dialogue_array[train_indices].tolist()
        holdout_ids = dialogue_array[holdout_indices].tolist()
        if set(train_ids) & set(holdout_ids):
            raise AssertionError("OOF train and holdout dialogues overlap")
        loaders = make_loaders(
            dataset, {"train": train_ids, "valid": holdout_ids, "test": []},
            args.batch_size, fold_seed, args.num_workers, device.type == "cuda")
        model = Transformer_Based_Model(**config).to(device)
        criterion = SDTCOLDLoss(
            class_weights(args, device), lambda_co=0.0, lambda_reg=0.0,
            lambda_reliability=0.0).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.l2)
        metrics = None
        for epoch in range(1, args.epochs + 1):
            metrics, _, _, _ = run_epoch(
                model, criterion, loaders["train"], device, optimizer,
                args.max_batches, args.grad_clip)
            if epoch == 1 or epoch == args.epochs or epoch % 10 == 0:
                print("OOF fold {}/{} epoch {}/{} loss={:.4f} F1={:.2f}".format(
                    fold_index, args.folds, epoch, args.epochs,
                    metrics["total"], metrics["weighted_f1"]), flush=True)
        fold_rows = collect_fold(model, loaders["valid"], device, fold_index,
                                 args.reliability_temperature)
        all_rows.extend(fold_rows)
        fold_metadata.append({
            "fold": fold_index, "seed": fold_seed,
            "train_dialogues": train_ids, "holdout_dialogues": holdout_ids,
            "final_train_metrics": metrics, "holdout_utterances": len(fold_rows),
        })

    dialogue_order = {key: index for index, key in enumerate(dataset.trainVid)}
    all_rows.sort(key=lambda row: (
        dialogue_order[row["dialogue_id"]], row["utterance_index"]))
    expected_count = sum(len(dataset.videoLabels[key]) for key in dataset.trainVid)
    keys = {(row["dialogue_id"], row["utterance_index"]) for row in all_rows}
    if len(all_rows) != expected_count or len(keys) != expected_count:
        if not args.max_batches:
            raise AssertionError("OOF manifest does not cover every training utterance")
        print("WARNING: --max-batches produced an incomplete debug manifest", flush=True)
    write_csv(output_path, all_rows)
    metadata = {
        "args": vars(args), "model_config": config,
        "feature_path": str(dataset.feature_path),
        "train_dialogues": list(dataset.trainVid),
        "test_dialogues_used": False,
        "expected_utterances": expected_count,
        "written_utterances": len(all_rows),
        "folds": fold_metadata,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    print("Wrote {} OOF targets to {}".format(len(all_rows), output_path), flush=True)
    print("The test split was not loaded into any OOF fold.", flush=True)
    return output_path


if __name__ == "__main__":
    main()
