"""Summarize Wheel mechanism controls and correct-Wheel reference runs."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from tical import circular_class_distance_matrix, emotion_wheel_angles


ORDER = ("B0", "P-0", "TRUE-P", "RAND-P", "TRUE-C", "RAND-C",
         "TRUE-PC", "RAND-PC", "DETACH-PC", "P-FIXED")


def identify(args):
    if not args.get("use_emotion_wheel", False):
        return "B0"
    proto = float(args.get("lambda_wheel_proto", 0)) > 0
    cpcc = float(args.get("lambda_wheel_cpcc", 0)) > 0
    if not proto and not cpcc:
        return "P-0"
    suffix = "PC" if proto and cpcc else "P" if proto else "C"
    if args.get("wheel_detach_features", False):
        return "DETACH-" + suffix
    if args.get("wheel_radius_mode", "free") == "fixed":
        return "P-FIXED" if suffix == "PC" else "FIXED-" + suffix
    prefix = "RAND" if args.get("wheel_semantic_control", "correct") == "random" else "TRUE"
    return prefix + "-" + suffix


def semantic_metrics(run_dir, dataset):
    path = run_dir / "test_predictions.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    labels = np.asarray([int(row["label"]) for row in rows])
    predictions = np.asarray([int(row["prediction"]) for row in rows])
    n_classes = 6 if dataset == "IEMOCAP" else 7
    distances = circular_class_distance_matrix(
        emotion_wheel_angles(dataset, n_classes)).numpy()
    errors = distances[labels, predictions]
    wrong = labels != predictions
    positive = sorted(float(x) for x in np.unique(distances) if x > 1e-7)
    adjacent, far = positive[0], positive[-1]
    wrong_count = max(int(wrong.sum()), 1)
    return {
        "semantic_error": float(errors.mean()),
        "semantic_error_wrong": float(errors[wrong].mean()) if wrong.any() else 0.0,
        "adjacent_error_pct": 100.0 * float(np.sum(wrong & np.isclose(errors, adjacent))) / wrong_count,
        "far_error_pct": 100.0 * float(np.sum(wrong & np.isclose(errors, far))) / wrong_count,
    }


def load_runs(roots):
    rows = []
    for root in roots:
        for summary_path in sorted(root.rglob("summary.json")):
            config_path = summary_path.parent / "config.json"
            if not config_path.is_file():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
            config = json.loads(config_path.read_text(encoding="utf-8-sig"))
            if summary.get("smoke_test", False):
                continue
            args, test = config["args"], summary["test"]
            run_id = identify(args)
            if run_id not in ORDER:
                continue
            row = {
                "id": run_id, "dataset": args["dataset"],
                "seed": int(args["seed"]), "best_epoch": int(summary["best_epoch"]),
                "weighted_f1": float(test["weighted_f1"]),
                "macro_f1": float(test.get("macro_f1", float("nan"))),
                "accuracy": float(test["accuracy"]),
                "run_dir": str(summary_path.parent.resolve()),
            }
            row.update(semantic_metrics(summary_path.parent, args["dataset"]))
            rows.append(row)
    seen = {}
    for row in rows:
        key = (row["id"], row["seed"])
        if key in seen:
            raise SystemExit("Duplicate run {}. Use clean result/reference directories.\n{}\n{}".format(
                key, seen[key], row["run_dir"]))
        seen[key] = row["run_dir"]
    return rows


def mean_std(values):
    values = [x for x in values if x == x]
    return (statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0)


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["id"]].append(row)
    output = []
    metrics = ("weighted_f1", "macro_f1", "accuracy", "semantic_error",
               "semantic_error_wrong", "adjacent_error_pct", "far_error_pct")
    for run_id in ORDER:
        if not groups[run_id]:
            continue
        item = {"id": run_id, "runs": len(groups[run_id])}
        for metric in metrics:
            mean, std = mean_std([row.get(metric, float("nan")) for row in groups[run_id]])
            item[metric + "_mean"], item[metric + "_std"] = mean, std
        output.append(item)
    return output


def paired(rows):
    lookup = {(row["id"], row["seed"]): row for row in rows}
    comparisons = (("TRUE-PC", "RAND-PC"), ("TRUE-P", "RAND-P"),
                   ("TRUE-C", "RAND-C"), ("TRUE-PC", "P-0"),
                   ("TRUE-PC", "DETACH-PC"), ("TRUE-PC", "P-FIXED"),
                   ("TRUE-PC", "B0"))
    output = []
    for first, second in comparisons:
        seeds = sorted(seed for run_id, seed in lookup
                       if run_id == first and (second, seed) in lookup)
        if not seeds:
            continue
        deltas = [lookup[(first, seed)]["weighted_f1"] -
                  lookup[(second, seed)]["weighted_f1"] for seed in seeds]
        sem = [lookup[(first, seed)].get("semantic_error", float("nan")) -
               lookup[(second, seed)].get("semantic_error", float("nan")) for seed in seeds]
        output.append({
            "comparison": first + " - " + second, "paired_seeds": len(seeds),
            "delta_weighted_f1_mean": mean_std(deltas)[0],
            "delta_weighted_f1_std": mean_std(deltas)[1],
            "delta_semantic_error_mean": mean_std(sem)[0],
            "wins": sum(x > 0 for x in deltas),
        })
    return output


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path,
                        help="control result dir followed by optional old reference dirs")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    roots = [path.resolve() for path in args.results]
    out = (args.output_dir or roots[0] / "mechanism_summary").resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = load_runs(roots)
    aggregates, deltas = aggregate(rows), paired(rows)
    write_csv(out / "mechanism_runs.csv", rows)
    write_csv(out / "mechanism_summary.csv", aggregates)
    write_csv(out / "mechanism_paired_deltas.csv", deltas)
    lines = ["# Emotion Wheel mechanism summary", "",
             "All checkpoints in this analysis were selected by each run's saved protocol.", "",
             "| ID | Runs | Weighted F1 | Semantic error | Far errors among errors |",
             "|---|---:|---:|---:|---:|"]
    for row in aggregates:
        lines.append("| {id} | {runs} | {weighted_f1_mean:.3f} ± {weighted_f1_std:.3f} | "
                     "{semantic_error_mean:.4f} ± {semantic_error_std:.4f} | "
                     "{far_error_pct_mean:.2f}% |".format(**row))
    lines += ["", "## Paired comparisons", "",
              "| Comparison | Seeds | Δ weighted F1 | Δ semantic error | Wins |",
              "|---|---:|---:|---:|---:|"]
    for row in deltas:
        lines.append("| {comparison} | {paired_seeds} | {delta_weighted_f1_mean:.3f} ± "
                     "{delta_weighted_f1_std:.3f} | {delta_semantic_error_mean:.4f} | "
                     "{wins}/{paired_seeds} |".format(**row))
    (out / "mechanism_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Read {} runs; wrote {}".format(len(rows), out))


if __name__ == "__main__":
    main()

