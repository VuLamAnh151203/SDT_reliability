"""Train/evaluate the standalone SDT + COLD variants on IEMOCAP or MELD."""

import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from dataloader import BASE_DIR, DialogueDataset, make_loaders, split_dialogues
from losses import SDTCOLDLoss
from model import (DISTRIBUTION_INIT_MODES, FUSION_VARIANTS, MODALITIES,
                   Transformer_Based_Model)
from reliability_data import OOFReliabilityTable


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--Dataset", "--dataset", dest="dataset", choices=("IEMOCAP", "MELD"), default="IEMOCAP")
    parser.add_argument("--feature-path")
    parser.add_argument("--fusion-variant", choices=FUSION_VARIANTS, default="guided",
                        help="guided=B, replace=A, oof-guided=OOF reliability, sdt=baseline")
    parser.add_argument("--distribution-init", choices=DISTRIBUTION_INIT_MODES,
                        default="random",
                        help="sdt-preserving initializes mu=H' and variance small")
    parser.add_argument("--initial-logvar", type=float, default=-6.0,
                        help="initial constant log-variance for sdt-preserving mode")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", "--hidden_dim", dest="hidden_dim", type=int, default=1024)
    parser.add_argument("--n-head", "--n_head", dest="n_head", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--l2", "--weight-decay", dest="l2", type=float, default=1e-5)
    parser.add_argument("--temp", "--temperature", dest="temp", type=float, default=1.0)
    parser.add_argument("--gamma-1", type=float, default=1.0)
    parser.add_argument("--gamma-2", type=float, default=1.0)
    parser.add_argument("--gamma-3", type=float, default=1.0)
    parser.add_argument("--lambda-co", type=float, default=0.1)
    parser.add_argument("--lambda-reg", type=float, default=0.1)
    parser.add_argument("--oof-reliability-targets",
                        help="CSV made by build_oof_reliability.py; required by oof-guided")
    parser.add_argument("--lambda-reliability", type=float, default=1.0)
    parser.add_argument("--modality-prune-quantile", type=float, default=0.0,
                        help="drop this lowest fraction per modality from student CE/KD")
    parser.add_argument("--sample-prune-quantile", type=float, default=0.0,
                        help="ignore this highest-noise fraction in training losses")
    parser.add_argument("--disagreement-weight", type=float, default=0.0,
                        help="add this times JSD to the OOF sample-noise score")
    parser.add_argument("--sample-prune-any", action="store_true",
                        help="allow sample pruning without requiring all modalities to be wrong")
    parser.add_argument("--logvar-min", type=float, default=-8.0)
    parser.add_argument("--logvar-max", type=float, default=8.0)
    parser.add_argument("--cold-eps", type=float, default=1e-8)
    parser.add_argument("--no-detach-errors", action="store_true",
                        help="also backpropagate COLD through per-utterance CE targets")
    parser.add_argument("--no-class-weight", action="store_true",
                        help="disable original IEMOCAP class weighting; COLD errors are always unweighted")
    parser.add_argument("--selection-protocol", choices=("test", "validation"), default="test",
                        help="test reproduces original SDT epoch selection; validation holds out train dialogues")
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=0, help="0 keeps the PyTorch default")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="0 disables clipping, as in SDT")
    parser.add_argument("--max-batches", type=int, default=0, help="smoke check: cap each split per epoch; 0=all")
    parser.add_argument("--output-dir", default=str(BASE_DIR / "results"))
    parser.add_argument("--eval-checkpoint", help="evaluate a saved checkpoint; architecture/loss settings come from it")
    return parser


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(args):
    if args.no_cuda or args.device == "cpu":
        return torch.device("cpu")
    if args.device == "auto" and not torch.cuda.is_available():
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    return torch.device("cuda", args.gpu_id)


def make_model_config(args, dataset):
    return {"dataset": args.dataset, "temp": args.temp, **dataset.feature_dims,
            "n_head": args.n_head, "n_classes": dataset.n_classes,
            "hidden_dim": args.hidden_dim, "n_speakers": dataset.n_speakers,
            "dropout": args.dropout, "fusion_variant": args.fusion_variant,
            "logvar_min": args.logvar_min, "logvar_max": args.logvar_max,
            "cold_eps": args.cold_eps, "distribution_init": args.distribution_init,
            "initial_logvar": args.initial_logvar}


def make_criterion(args, device):
    weights = None
    if args.dataset == "IEMOCAP" and not args.no_class_weight:
        weights = torch.tensor([1 / p for p in (0.086747, 0.144406, 0.227883,
                                               0.160585, 0.127711, 0.252668)], device=device)
    return SDTCOLDLoss(weights, args.gamma_1, args.gamma_2, args.gamma_3,
                       args.lambda_co, args.lambda_reg, not args.no_detach_errors,
                       args.lambda_reliability).to(device)


def run_epoch(model, criterion, loader, device, optimizer=None, max_batches=0,
              grad_clip=0.0, collect_predictions=False,
              reliability_table=None):
    training = optimizer is not None
    model.train(training)
    totals, true, predicted, rows = {}, [], [], []
    count = 0
    with torch.set_grad_enabled(training):
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            text, visual, audio, speakers, mask, labels = [item.to(device) for item in batch[:6]]
            valid = mask.bool()
            lengths = valid.sum(dim=1).tolist()
            reliability_targets = sample_keep = modality_keep = None
            if training and reliability_table is not None:
                reliability_targets, sample_keep, modality_keep = reliability_table.batch(
                    batch[6], lengths, mask.size(1), device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            output = model(text, visual, audio, mask, speakers.transpose(0, 1), lengths)
            loss, parts, errors = criterion(
                output, labels, valid, reliability_targets,
                sample_keep, modality_keep)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss in batch {}".format(batch_index))
            if training:
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            n_valid = int(valid.sum().item())
            count += n_valid
            for name, value in parts.items():
                totals[name] = totals.get(name, 0.0) + value.detach().item() * n_valid
            prediction = output["logits"].argmax(dim=-1)
            true.extend(labels[valid].cpu().tolist())
            predicted.extend(prediction[valid].cpu().tolist())
            if collect_predictions:
                rows.extend(prediction_rows(batch[6], lengths, labels, prediction, output, errors))
    if not count:
        raise ValueError("no valid utterances in the data loader")
    metrics = {name: value / count for name, value in totals.items()}
    metrics.update({"accuracy": 100.0 * accuracy_score(true, predicted),
                    "weighted_f1": 100.0 * f1_score(true, predicted, average="weighted", zero_division=0),
                    "utterances": count})
    return metrics, rows, true, predicted


def prediction_rows(dialogue_ids, lengths, labels, prediction, output, errors):
    labels, prediction = labels.cpu(), prediction.cpu()
    probabilities = output["prob"].detach().cpu()
    gate = output["fusion_weights"].detach().mean(dim=-1).cpu()
    reliability = output["reliability"]
    variance_norm = output["variance_norm"]
    if reliability is not None:
        reliability = reliability.detach().cpu()
    if variance_norm is not None:
        variance_norm = variance_norm.detach().cpu()
    students = {m: output["student_logits"][m].detach().argmax(dim=-1).cpu() for m in MODALITIES}
    errors = {m: value.detach().cpu() for m, value in errors.items()}
    offset = 0
    rows = []
    for batch_index, (dialogue_id, length) in enumerate(zip(dialogue_ids, lengths)):
        for index in range(length):
            row = {"dialogue_id": dialogue_id, "utterance_index": index,
                   "label": int(labels[batch_index, index]), "prediction": int(prediction[batch_index, index])}
            row.update({"prob_{}".format(c): float(p) for c, p in enumerate(probabilities[batch_index, index])})
            for modal_index, name in enumerate(MODALITIES):
                row[name + "_prediction"] = int(students[name][batch_index, index])
                row[name + "_gate_mean"] = float(gate[batch_index, index, modal_index])
                if reliability is not None:
                    row[name + "_reliability"] = float(reliability[batch_index, index, modal_index])
                if variance_norm is not None:
                    row[name + "_variance_norm"] = float(variance_norm[batch_index, index, modal_index])
                if reliability is not None:
                    if name in errors:
                        row[name + "_ce_error"] = float(errors[name][offset])
                    else:
                        label = int(labels[batch_index, index])
                        row[name + "_ce_error"] = float(
                            -output["student_log_prob"][name][batch_index, index, label].detach().cpu())
            rows.append(row)
            offset += 1
    return rows


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def save_test_outputs(run_dir, metrics, rows, true, predicted, n_classes):
    write_json(run_dir / "test_metrics.json", metrics)
    write_csv(run_dir / "test_predictions.csv", rows)
    write_json(run_dir / "classification_report.json", classification_report(
        true, predicted, labels=list(range(n_classes)), output_dict=True, zero_division=0))
    write_json(run_dir / "confusion_matrix.json", confusion_matrix(
        true, predicted, labels=list(range(n_classes))).tolist())


def main(argv=None):
    args = build_parser().parse_args(argv)
    if min(args.epochs, args.batch_size) < 1 or min(args.num_workers, args.max_batches, args.num_threads) < 0:
        raise ValueError("epochs/batch size must be positive; worker/thread/batch limits must be nonnegative")
    if args.lr <= 0 or args.l2 < 0 or args.grad_clip < 0:
        raise ValueError("lr must be positive; l2 and grad_clip must be nonnegative")
    if args.lambda_reliability < 0 or args.disagreement_weight < 0:
        raise ValueError("reliability/disagreement weights must be nonnegative")
    for name in ("modality_prune_quantile", "sample_prune_quantile"):
        if not 0 <= getattr(args, name) < 1:
            raise ValueError("{} must be in [0, 1)".format(name))
    if args.fusion_variant == "oof-guided" and not args.oof_reliability_targets:
        raise ValueError("--fusion-variant oof-guided requires --oof-reliability-targets")
    if args.fusion_variant != "oof-guided" and args.oof_reliability_targets:
        raise ValueError("--oof-reliability-targets requires --fusion-variant oof-guided")
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    seed_everything(args.seed)
    device = resolve_device(args)
    checkpoint = None
    if args.eval_checkpoint:
        checkpoint = torch.load(args.eval_checkpoint, map_location="cpu", weights_only=True)
        args.dataset = checkpoint["model_config"]["dataset"]
        for name in ("temp", "n_head", "hidden_dim", "dropout", "fusion_variant",
                     "logvar_min", "logvar_max", "cold_eps", "distribution_init",
                     "initial_logvar"):
            if name in checkpoint["model_config"]:
                setattr(args, name, checkpoint["model_config"][name])
        for name in ("gamma_1", "gamma_2", "gamma_3", "lambda_co", "lambda_reg",
                     "no_detach_errors", "no_class_weight", "selection_protocol", "valid_ratio",
                     "lambda_reliability", "modality_prune_quantile",
                     "sample_prune_quantile", "disagreement_weight", "sample_prune_any",
                     "oof_reliability_targets"):
            if name in checkpoint["args"]:
                setattr(args, name, checkpoint["args"][name])
        if args.feature_path is None:
            saved_path = checkpoint["feature_path"]
            if Path(saved_path).is_file():
                args.feature_path = saved_path
    dataset = DialogueDataset(args.dataset, args.feature_path)
    split_ids = split_dialogues(dataset, args.selection_protocol, args.valid_ratio)
    reliability_table = None
    if args.fusion_variant == "oof-guided" and not args.eval_checkpoint:
        reliability_table = OOFReliabilityTable(
            args.oof_reliability_targets, args.modality_prune_quantile,
            args.sample_prune_quantile, args.disagreement_weight,
            args.sample_prune_any)
        reliability_table.validate_dialogues(dataset, split_ids["train"])
    model_config = checkpoint["model_config"] if checkpoint else make_model_config(args, dataset)
    if any(model_config[k] != v for k, v in dataset.feature_dims.items()):
        raise ValueError("checkpoint feature dimensions do not match the dataset")
    if checkpoint and split_ids != checkpoint["split_ids"]:
        raise ValueError("checkpoint dialogue split differs from the supplied feature pickle")
    model = Transformer_Based_Model(**model_config).to(device)
    criterion = make_criterion(args, device)
    loaders = make_loaders(dataset, split_ids, args.batch_size, args.seed, args.num_workers, device.type == "cuda")
    if model_config["fusion_variant"] in ("guided", "replace"):
        init_tag = model_config.get("distribution_init", "random").replace("-", "_")
    else:
        init_tag = "no_distribution"
    run_name = "{}_{}_{}_seed{}_{}".format(
        args.dataset.lower(), model_config["fusion_variant"], init_tag, args.seed,
        datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_dir = Path(args.output_dir).resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", {"args": vars(args), "model_config": model_config,
                                        "device": str(device), "feature_path": str(dataset.feature_path),
                                        "reliability_targets": (reliability_table.summary()
                                                                if reliability_table else None),
                                        "smoke_test": args.max_batches > 0})
    write_json(run_dir / "split_ids.json", split_ids)
    print("Device: {}; variant: {}; distribution init: {}; selection: {}; output: {}".format(
        device, model_config["fusion_variant"], init_tag,
        args.selection_protocol, run_dir), flush=True)
    print("Dialogues: {}; parameters: {:,}".format(
        {name: len(ids) for name, ids in split_ids.items()}, sum(p.numel() for p in model.parameters())), flush=True)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        metrics, rows, true, predicted = run_epoch(model, criterion, loaders["test"], device,
                                                  max_batches=args.max_batches, collect_predictions=True)
        save_test_outputs(run_dir, metrics, rows, true, predicted, dataset.n_classes)
        print(json.dumps(metrics, indent=2))
        return run_dir

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    best_score, best_epoch = -1.0, 0
    history = []
    selection_split = "test" if args.selection_protocol == "test" else "valid"
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_metrics, _, _, _ = run_epoch(model, criterion, loaders["train"], device, optimizer,
                                           args.max_batches, args.grad_clip,
                                           reliability_table=reliability_table)
        selected_metrics, _, _, _ = run_epoch(model, criterion, loaders[selection_split], device,
                                              max_batches=args.max_batches)
        row = {"epoch": epoch, "seconds": time.time() - started}
        row.update({"train_" + k: v for k, v in train_metrics.items()})
        row.update({selection_split + "_" + k: v for k, v in selected_metrics.items()})
        history.append(row)
        write_csv(run_dir / "epoch_metrics.csv", history)
        if selected_metrics["weighted_f1"] > best_score:
            best_score, best_epoch = selected_metrics["weighted_f1"], epoch
            torch.save({"model_state_dict": model.state_dict(), "model_config": model_config,
                        "args": vars(args), "epoch": epoch, "selection_metrics": selected_metrics,
                        "split_ids": split_ids, "feature_path": str(dataset.feature_path)},
                       run_dir / "best_checkpoint.pt")
        print("Epoch {:03d} train loss={:.4f} F1={:.2f}; {} F1={:.2f}; "
              "COLD={:.4f} (weighted={:.4f}) reg={:.4f} (weighted={:.4f}) "
              "reliability={:.4f} (weighted={:.4f}) keep(sample/modality)={:.3f}/{:.3f} ({:.1f}s)".format(
            epoch, train_metrics["total"], train_metrics["weighted_f1"], selection_split,
            selected_metrics["weighted_f1"], train_metrics["cold"],
            train_metrics["weighted_cold"], train_metrics["reg"],
            train_metrics["weighted_reg"], train_metrics["reliability_loss"],
            train_metrics["weighted_reliability"], train_metrics["sample_keep_rate"],
            train_metrics["modality_keep_rate"], row["seconds"]), flush=True)
    checkpoint = torch.load(run_dir / "best_checkpoint.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics, rows, true, predicted = run_epoch(model, criterion, loaders["test"], device,
                                              max_batches=args.max_batches, collect_predictions=True)
    save_test_outputs(run_dir, metrics, rows, true, predicted, dataset.n_classes)
    write_json(run_dir / "summary.json", {"best_epoch": best_epoch, "selection_protocol": args.selection_protocol,
                                         "selection_weighted_f1": best_score, "test": metrics,
                                         "smoke_test": args.max_batches > 0})
    print("Best epoch: {}; test accuracy={:.2f}, weighted F1={:.2f}".format(
        best_epoch, metrics["accuracy"], metrics["weighted_f1"]), flush=True)
    return run_dir


if __name__ == "__main__":
    main()
