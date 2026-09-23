"""Train/evaluate SDT with optional COLD, OOF reliability, or TiCAL."""

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
    parser.add_argument("--use-tical", action="store_true",
                        help="enable TiCAL on the deterministic SDT path (incompatible with COLD variants)")
    parser.add_argument("--tical-mode", choices=("observe", "kd", "hyp", "fusion", "full"),
                        default="kd", help="incremental TiCAL ablation")
    parser.add_argument("--tical-warmup-epochs", type=int, default=5)
    parser.add_argument("--anchor-size", type=int, default=2048)
    parser.add_argument("--anchor-balance", choices=("none", "equal"), default="none",
                        help="none=global FIFO; equal=separate equal-capacity FIFO per class")
    parser.add_argument("--anchor-min-per-class", type=int, default=0,
                        help="query only after every class in every bank has N anchors; 0 keeps old readiness")
    parser.add_argument("--anchor-admission", choices=("teacher", "modality"),
                        default="teacher",
                        help="modality also requires the matching unimodal classifier to be correct")
    parser.add_argument("--anchor-conf-threshold", type=float, default=0.8)
    parser.add_argument("--hyperbolic-dim", type=int, default=128)
    parser.add_argument("--hyp-eps", type=float, default=1e-5)
    parser.add_argument("--typicality-eps", type=float, default=1e-8)
    parser.add_argument("--consistency-t", type=float, default=0.2)
    parser.add_argument("--consistency-k", type=float, default=0.5)
    parser.add_argument("--no-detach-tau", action="store_true")
    parser.add_argument("--no-detach-kappa", action="store_true")
    parser.add_argument("--beta-gate", type=float, default=1.0)
    parser.add_argument("--lambda-hyp", type=float, default=0.1)
    parser.add_argument("--use-emotion-wheel", action="store_true",
                        help="constrain TiCAL with fixed hyperbolic emotion-wheel prototypes")
    parser.add_argument("--wheel-prototype-radius", type=float, default=0.75)
    parser.add_argument("--wheel-temperature", type=float, default=1.0)
    parser.add_argument("--wheel-anchor-mix", type=float, default=0.5,
                        help="0=anchor typicality, 1=prototype typicality")
    parser.add_argument("--lambda-wheel-proto", type=float, default=0.0)
    parser.add_argument("--lambda-wheel-cpcc", type=float, default=0.0)
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
            "initial_logvar": args.initial_logvar, "use_tical": args.use_tical,
            "tical_mode": args.tical_mode,
            "tical_warmup_epochs": args.tical_warmup_epochs,
            "anchor_size": args.anchor_size,
            "anchor_balance": args.anchor_balance,
            "anchor_min_per_class": args.anchor_min_per_class,
            "anchor_admission": args.anchor_admission,
            "anchor_conf_threshold": args.anchor_conf_threshold,
            "hyperbolic_dim": args.hyperbolic_dim, "hyp_eps": args.hyp_eps,
            "typicality_eps": args.typicality_eps,
            "consistency_t": args.consistency_t,
            "consistency_k": args.consistency_k,
            "detach_tau": not args.no_detach_tau,
            "detach_kappa": not args.no_detach_kappa,
            "beta_gate": args.beta_gate,
            "use_emotion_wheel": args.use_emotion_wheel,
            "wheel_prototype_radius": args.wheel_prototype_radius,
            "wheel_temperature": args.wheel_temperature,
            "wheel_anchor_mix": args.wheel_anchor_mix}


def make_criterion(args, device):
    weights = None
    if args.dataset == "IEMOCAP" and not args.no_class_weight:
        weights = torch.tensor([1 / p for p in (0.086747, 0.144406, 0.227883,
                                               0.160585, 0.127711, 0.252668)], device=device)
    return SDTCOLDLoss(weights, args.gamma_1, args.gamma_2, args.gamma_3,
                       args.lambda_co, args.lambda_reg, not args.no_detach_errors,
                       args.lambda_reliability,
                       args.tical_mode if args.use_tical else None,
                       args.lambda_hyp, args.lambda_wheel_proto,
                       args.lambda_wheel_cpcc).to(device)


def _collect_tical_batch(storage, output, labels, prediction, valid):
    tical = output.get("tical")
    if tical is None or not tical["ready"]:
        return
    for name in MODALITIES:
        storage["tau_" + name].append(tical["tau"][name].detach().cpu())
        storage["pseudo_" + name].append(tical["pseudo_labels"][name].detach().cpu())
        if tical.get("anchor_tau") is not None:
            storage["anchor_tau_" + name].append(
                tical["anchor_tau"][name].detach().cpu())
        if tical.get("prototype_tau") is not None:
            storage["prototype_tau_" + name].append(
                tical["prototype_tau"][name].detach().cpu())
        if tical.get("wheel_pseudo_labels") is not None:
            storage["wheel_pseudo_" + name].append(
                tical["wheel_pseudo_labels"][name].detach().cpu())
    storage["kappa"].append(tical["kappa"].detach().cpu())
    storage["categorical_discrepancy"].append(
        tical["categorical_discrepancy"].detach().cpu())
    if tical.get("wheel_discrepancy") is not None:
        storage["wheel_discrepancy"].append(
            tical["wheel_discrepancy"].detach().cpu())
    storage["correct"].append(prediction[valid].eq(labels[valid]).detach().cpu())


def _tical_epoch_metrics(storage, model, utterance_count):
    if not getattr(model, "use_tical", False):
        return {}
    result = {"tical_ready_rate": 0.0, "anchors_added": float(storage["anchors_added"]),
              "agreement_ta": 0.0, "agreement_tv": 0.0,
              "agreement_av": 0.0, "agreement_tav": 0.0,
              "kappa_mean": 0.0, "kappa_std": 0.0,
              "kappa_q05": 0.0, "kappa_q50": 0.0, "kappa_q95": 0.0}
    for name in MODALITIES:
        for statistic in ("mean", "std", "q05", "q50", "q95"):
            result["tau_{}_{}".format(name, statistic)] = 0.0
        if getattr(model, "use_emotion_wheel", False):
            result["anchor_tau_{}_mean".format(name)] = 0.0
            result["prototype_tau_{}_mean".format(name)] = 0.0
            result["wheel_pseudo_{}_anchor_agreement".format(name)] = 0.0
    if getattr(model, "use_emotion_wheel", False):
        result["categorical_discrepancy_mean"] = 0.0
        result["wheel_discrepancy_mean"] = 0.0
    for group in ("low", "medium", "high"):
        result["kappa_{}_frac".format(group)] = 0.0
        result["kappa_{}_accuracy".format(group)] = 0.0
    summary = model.tical_anchor_summary()
    for name in MODALITIES:
        result["anchor_{}_size".format(name)] = float(summary[name]["size"])
        for class_index, count in enumerate(summary[name]["class_counts"]):
            result["anchor_{}_class_{}".format(name, class_index)] = float(count)
    if not storage["kappa"]:
        return result
    values = {key: torch.cat(chunks) for key, chunks in storage.items()
              if isinstance(chunks, list) and chunks}
    n_ready = int(values["kappa"].numel())
    result["tical_ready_rate"] = n_ready / max(1, utterance_count)
    for name in MODALITIES:
        tau = values["tau_" + name].float()
        result.update({
            "tau_{}_mean".format(name): tau.mean().item(),
            "tau_{}_std".format(name): tau.std(unbiased=False).item(),
            "tau_{}_q05".format(name): torch.quantile(tau, 0.05).item(),
            "tau_{}_q50".format(name): torch.quantile(tau, 0.50).item(),
            "tau_{}_q95".format(name): torch.quantile(tau, 0.95).item(),
        })
        if getattr(model, "use_emotion_wheel", False):
            result["anchor_tau_{}_mean".format(name)] = values[
                "anchor_tau_" + name].float().mean().item()
            result["prototype_tau_{}_mean".format(name)] = values[
                "prototype_tau_" + name].float().mean().item()
            result["wheel_pseudo_{}_anchor_agreement".format(name)] = (
                values["wheel_pseudo_" + name]
                .eq(values["pseudo_" + name]).float().mean().item())
    if getattr(model, "use_emotion_wheel", False):
        result["categorical_discrepancy_mean"] = values[
            "categorical_discrepancy"].float().mean().item()
        result["wheel_discrepancy_mean"] = values[
            "wheel_discrepancy"].float().mean().item()
    pseudo_t, pseudo_a, pseudo_v = (values["pseudo_" + name] for name in MODALITIES)
    result.update({
        "agreement_ta": pseudo_t.eq(pseudo_a).float().mean().item(),
        "agreement_tv": pseudo_t.eq(pseudo_v).float().mean().item(),
        "agreement_av": pseudo_a.eq(pseudo_v).float().mean().item(),
        "agreement_tav": (pseudo_t.eq(pseudo_a) & pseudo_t.eq(pseudo_v)).float().mean().item(),
    })
    kappa, correct = values["kappa"].float(), values["correct"].float()
    result.update({
        "kappa_mean": kappa.mean().item(),
        "kappa_std": kappa.std(unbiased=False).item(),
        "kappa_q05": torch.quantile(kappa, 0.05).item(),
        "kappa_q50": torch.quantile(kappa, 0.50).item(),
        "kappa_q95": torch.quantile(kappa, 0.95).item(),
    })
    groups = {"low": kappa < 0.3,
              "medium": (kappa >= 0.3) & (kappa < 0.7),
              "high": kappa >= 0.7}
    for group, selected in groups.items():
        result["kappa_{}_frac".format(group)] = selected.float().mean().item()
        result["kappa_{}_accuracy".format(group)] = (
            correct[selected].mean().item() if selected.any() else 0.0)
    return result


def run_epoch(model, criterion, loader, device, optimizer=None, max_batches=0,
              grad_clip=0.0, collect_predictions=False,
              reliability_table=None, epoch=0):
    training = optimizer is not None
    model.train(training)
    model.set_tical_epoch(epoch)
    totals, true, predicted, rows = {}, [], [], []
    tical_storage = {"tau_" + name: [] for name in MODALITIES}
    tical_storage.update({"pseudo_" + name: [] for name in MODALITIES})
    tical_storage.update({"anchor_tau_" + name: [] for name in MODALITIES})
    tical_storage.update({"prototype_tau_" + name: [] for name in MODALITIES})
    tical_storage.update({"wheel_pseudo_" + name: [] for name in MODALITIES})
    tical_storage.update({"kappa": [], "correct": [],
                          "categorical_discrepancy": [],
                          "wheel_discrepancy": [], "anchors_added": 0})
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
                # Query happened in forward against OLD banks. Insert the
                # current detached batch only after the parameter update.
                tical_storage["anchors_added"] += model.update_tical_anchors(
                    output, labels, valid)
            n_valid = int(valid.sum().item())
            count += n_valid
            for name, value in parts.items():
                totals[name] = totals.get(name, 0.0) + value.detach().item() * n_valid
            prediction = output["logits"].argmax(dim=-1)
            _collect_tical_batch(tical_storage, output, labels, prediction, valid)
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
    metrics.update(_tical_epoch_metrics(tical_storage, model, count))
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
    tical = output.get("tical")
    tical_ready = bool(tical is not None and tical["ready"])
    if tical_ready:
        tical = {
            "kappa": tical["kappa"].detach().cpu(),
            "label_discrepancy": tical["label_discrepancy"].detach().cpu(),
            "categorical_discrepancy": tical[
                "categorical_discrepancy"].detach().cpu(),
            "wheel_discrepancy": (
                tical["wheel_discrepancy"].detach().cpu()
                if tical.get("wheel_discrepancy") is not None else None),
            "tau": {m: tical["tau"][m].detach().cpu() for m in MODALITIES},
            "anchor_tau": {
                m: tical["anchor_tau"][m].detach().cpu() for m in MODALITIES},
            "prototype_tau": (
                {m: tical["prototype_tau"][m].detach().cpu()
                 for m in MODALITIES}
                if tical.get("prototype_tau") is not None else None),
            "pseudo": {m: tical["pseudo_labels"][m].detach().cpu() for m in MODALITIES},
            "wheel_pseudo": (
                {m: tical["wheel_pseudo_labels"][m].detach().cpu()
                 for m in MODALITIES}
                if tical.get("wheel_pseudo_labels") is not None else None),
        }
    offset = 0
    rows = []
    for batch_index, (dialogue_id, length) in enumerate(zip(dialogue_ids, lengths)):
        for index in range(length):
            row = {"dialogue_id": dialogue_id, "utterance_index": index,
                   "label": int(labels[batch_index, index]), "prediction": int(prediction[batch_index, index])}
            if tical_ready:
                row["kappa"] = float(tical["kappa"][offset])
                row["label_discrepancy"] = float(tical["label_discrepancy"][offset])
                row["categorical_discrepancy"] = float(
                    tical["categorical_discrepancy"][offset])
                if tical["wheel_discrepancy"] is not None:
                    row["wheel_discrepancy"] = float(
                        tical["wheel_discrepancy"][offset])
            row.update({"prob_{}".format(c): float(p) for c, p in enumerate(probabilities[batch_index, index])})
            for modal_index, name in enumerate(MODALITIES):
                row[name + "_prediction"] = int(students[name][batch_index, index])
                row[name + "_gate_mean"] = float(gate[batch_index, index, modal_index])
                if tical_ready:
                    row[name + "_tau"] = float(tical["tau"][name][offset])
                    row[name + "_anchor_tau"] = float(
                        tical["anchor_tau"][name][offset])
                    row[name + "_pseudo_label"] = int(
                        tical["pseudo"][name][offset])
                    if tical["prototype_tau"] is not None:
                        row[name + "_prototype_tau"] = float(
                            tical["prototype_tau"][name][offset])
                    if tical["wheel_pseudo"] is not None:
                        row[name + "_wheel_pseudo_label"] = int(
                            tical["wheel_pseudo"][name][offset])
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
    if (args.tical_warmup_epochs < 0 or args.anchor_size < 1
            or args.anchor_min_per_class < 0
            or args.hyperbolic_dim < 1 or args.lambda_hyp < 0
            or args.beta_gate < 0 or args.consistency_t < 0
            or args.consistency_k < 0 or args.lambda_wheel_proto < 0
            or args.lambda_wheel_cpcc < 0):
        raise ValueError("invalid nonnegative TiCAL setting")
    if not 0 <= args.anchor_conf_threshold <= 1:
        raise ValueError("--anchor-conf-threshold must be in [0,1]")
    for name in ("modality_prune_quantile", "sample_prune_quantile"):
        if not 0 <= getattr(args, name) < 1:
            raise ValueError("{} must be in [0, 1)".format(name))
    if args.fusion_variant == "oof-guided" and not args.oof_reliability_targets:
        raise ValueError("--fusion-variant oof-guided requires --oof-reliability-targets")
    if args.fusion_variant != "oof-guided" and args.oof_reliability_targets:
        raise ValueError("--oof-reliability-targets requires --fusion-variant oof-guided")
    if args.use_tical and args.fusion_variant != "sdt":
        raise ValueError("--use-tical requires --fusion-variant sdt (COLD is disabled)")
    if args.use_emotion_wheel and not args.use_tical:
        raise ValueError("--use-emotion-wheel requires --use-tical")
    if not args.use_emotion_wheel and (
            args.lambda_wheel_proto > 0 or args.lambda_wheel_cpcc > 0):
        raise ValueError("wheel loss weights require --use-emotion-wheel")
    if args.use_emotion_wheel:
        if args.hyperbolic_dim < 2:
            raise ValueError("emotion wheel requires --hyperbolic-dim >= 2")
        if not 0 < args.wheel_prototype_radius < 1 - args.hyp_eps:
            raise ValueError("--wheel-prototype-radius must be in (0, 1-hyp-eps)")
        if args.wheel_temperature <= 0:
            raise ValueError("--wheel-temperature must be positive")
        if not 0 <= args.wheel_anchor_mix <= 1:
            raise ValueError("--wheel-anchor-mix must be in [0,1]")
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
                     "initial_logvar", "use_tical", "tical_mode",
                     "tical_warmup_epochs", "anchor_size", "anchor_balance",
                     "anchor_min_per_class", "anchor_admission",
                     "anchor_conf_threshold",
                     "hyperbolic_dim", "hyp_eps", "typicality_eps",
                     "consistency_t", "consistency_k", "beta_gate",
                     "use_emotion_wheel", "wheel_prototype_radius",
                     "wheel_temperature", "wheel_anchor_mix"):
            if name in checkpoint["model_config"]:
                setattr(args, name, checkpoint["model_config"][name])
        for name in ("gamma_1", "gamma_2", "gamma_3", "lambda_co", "lambda_reg",
                     "no_detach_errors", "no_class_weight", "selection_protocol", "valid_ratio",
                     "lambda_reliability", "modality_prune_quantile",
                     "sample_prune_quantile", "disagreement_weight", "sample_prune_any",
                     "oof_reliability_targets", "tical_mode", "lambda_hyp",
                     "no_detach_tau", "no_detach_kappa", "use_emotion_wheel",
                     "wheel_prototype_radius", "wheel_temperature",
                     "wheel_anchor_mix", "lambda_wheel_proto",
                     "lambda_wheel_cpcc"):
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
    method_tag = ("tical_" + model_config["tical_mode"]
                  if model_config.get("use_tical") else model_config["fusion_variant"])
    if (model_config.get("use_tical")
            and model_config.get("anchor_balance", "none") != "none"):
        method_tag += "_anchors_" + model_config["anchor_balance"]
    if (model_config.get("use_tical")
            and model_config.get("anchor_admission", "teacher") != "teacher"):
        method_tag += "_admit_" + model_config["anchor_admission"]
    if (model_config.get("use_tical")
            and model_config.get("anchor_min_per_class", 0) > 0):
        method_tag += "_minclass{}".format(
            model_config["anchor_min_per_class"])
    if model_config.get("use_emotion_wheel"):
        method_tag += "_wheel"
    run_name = "{}_{}_{}_seed{}_{}".format(
        args.dataset.lower(), method_tag, init_tag, args.seed,
        datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_dir = Path(args.output_dir).resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", {"args": vars(args), "model_config": model_config,
                                        "device": str(device), "feature_path": str(dataset.feature_path),
                                        "reliability_targets": (reliability_table.summary()
                                                                if reliability_table else None),
                                        "smoke_test": args.max_batches > 0})
    write_json(run_dir / "split_ids.json", split_ids)
    print("Device: {}; method: {}; distribution init: {}; selection: {}; output: {}".format(
        device, method_tag, init_tag,
        args.selection_protocol, run_dir), flush=True)
    print("Dialogues: {}; parameters: {:,}".format(
        {name: len(ids) for name, ids in split_ids.items()}, sum(p.numel() for p in model.parameters())), flush=True)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        model.set_tical_epoch(checkpoint.get("epoch", 0))
        metrics, rows, true, predicted = run_epoch(model, criterion, loaders["test"], device,
                                                  max_batches=args.max_batches, collect_predictions=True,
                                                  epoch=checkpoint.get("epoch", 0))
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
                                           reliability_table=reliability_table, epoch=epoch)
        selected_metrics, _, _, _ = run_epoch(model, criterion, loaders[selection_split], device,
                                              max_batches=args.max_batches, epoch=epoch)
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
        if args.use_tical:
            print("Epoch {:03d} train loss={:.4f} F1={:.2f}; {} F1={:.2f}; "
                  "KL(original/CA)={:.4f}/{:.4f} hyp={:.4f}; "
                  "wheel(proto/cpcc)={:.4f}/{:.4f}; "
                  "kappa={:.3f} ready={:.3f}; anchors(T/A/V)={:.0f}/{:.0f}/{:.0f} ({:.1f}s)".format(
                epoch, train_metrics["total"], train_metrics["weighted_f1"], selection_split,
                selected_metrics["weighted_f1"], train_metrics["original_distillation"],
                train_metrics["ca_distillation"], train_metrics["weighted_hyp"],
                train_metrics["weighted_wheel_proto"],
                train_metrics["weighted_wheel_cpcc"],
                train_metrics.get("kappa_mean", 0.0), train_metrics["tical_ready_rate"],
                train_metrics["anchor_t_size"], train_metrics["anchor_a_size"],
                train_metrics["anchor_v_size"], row["seconds"]), flush=True)
        else:
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
    model.set_tical_epoch(checkpoint.get("epoch", best_epoch))
    metrics, rows, true, predicted = run_epoch(model, criterion, loaders["test"], device,
                                              max_batches=args.max_batches, collect_predictions=True,
                                              epoch=checkpoint.get("epoch", best_epoch))
    save_test_outputs(run_dir, metrics, rows, true, predicted, dataset.n_classes)
    write_json(run_dir / "summary.json", {"best_epoch": best_epoch, "selection_protocol": args.selection_protocol,
                                         "selection_weighted_f1": best_score, "test": metrics,
                                         "smoke_test": args.max_batches > 0})
    print("Best epoch: {}; test accuracy={:.2f}, weighted F1={:.2f}".format(
        best_epoch, metrics["accuracy"], metrics["weighted_f1"]), flush=True)
    return run_dir


if __name__ == "__main__":
    main()
