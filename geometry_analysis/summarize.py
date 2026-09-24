"""Aggregate controlled geometry ablation runs into CSV and Markdown."""

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ORDER = ("B0", "E-P", "E-C", "E-PC", "S-P", "S-C", "S-PC",
         "P-P", "P-C", "P-PC")


def experiment_id(args):
    if not args.get("use_emotion_wheel", False):
        return "B0"
    prefix = {"euclidean": "E", "spherical": "S", "poincare": "P"}[
        args.get("wheel_geometry", "poincare")]
    has_proto = float(args.get("lambda_wheel_proto", 0.0)) > 0.0
    has_cpcc = float(args.get("lambda_wheel_cpcc", 0.0)) > 0.0
    if not has_proto and not has_cpcc:
        return prefix + "-0"
    suffix = "PC" if has_proto and has_cpcc else "P" if has_proto else "C"
    return prefix + "-" + suffix


def load_runs(root, include_smoke=False):
    rows = []
    for summary_path in sorted(root.rglob("summary.json")):
        config_path = summary_path.parent / "config.json"
        if not config_path.is_file():
            continue
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        if summary.get("smoke_test", False) and not include_smoke:
            continue
        args = config["args"]
        run_id = experiment_id(args)
        if run_id not in ORDER:
            continue
        test = summary["test"]
        rows.append({
            "id": run_id,
            "dataset": args["dataset"],
            "seed": int(args["seed"]),
            "geometry": (args.get("wheel_geometry", "poincare")
                         if run_id != "B0" else "none"),
            "lambda_proto": float(args.get("lambda_wheel_proto", 0.0)),
            "lambda_cpcc": float(args.get("lambda_wheel_cpcc", 0.0)),
            "wheel_temperature": float(args.get("wheel_temperature", 1.0)),
            "best_epoch": int(summary["best_epoch"]),
            "selection_weighted_f1": float(summary["selection_weighted_f1"]),
            "test_accuracy": float(test["accuracy"]),
            "test_weighted_f1": float(test["weighted_f1"]),
            "test_macro_f1": float(test.get("macro_f1", float("nan"))),
            "wheel_t_accuracy": float(test.get("wheel_t_accuracy", float("nan"))),
            "wheel_a_accuracy": float(test.get("wheel_a_accuracy", float("nan"))),
            "wheel_v_accuracy": float(test.get("wheel_v_accuracy", float("nan"))),
            "run_dir": str(summary_path.parent.resolve()),
        })
    return rows


def mean_std(values):
    clean = [value for value in values if value == value]
    if not clean:
        return float("nan"), float("nan")
    return statistics.mean(clean), statistics.stdev(clean) if len(clean) > 1 else 0.0


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["id"]].append(row)
    result = []
    for run_id in ORDER:
        group = grouped.get(run_id, [])
        if not group:
            continue
        item = {"id": run_id, "runs": len(group)}
        for metric in ("selection_weighted_f1", "test_accuracy",
                       "test_weighted_f1", "test_macro_f1",
                       "wheel_t_accuracy", "wheel_a_accuracy",
                       "wheel_v_accuracy"):
            mean, std = mean_std([row[metric] for row in group])
            item[metric + "_mean"] = mean
            item[metric + "_std"] = std
        result.append(item)
    return result


def paired_deltas(rows):
    lookup = {(row["id"], row["seed"]): row for row in rows}
    comparisons = (
        ("P-PC", "S-PC"), ("P-PC", "E-PC"),
        ("P-P", "S-P"), ("P-C", "S-C"),
        ("E-PC", "B0"), ("S-PC", "B0"), ("P-PC", "B0"),
    )
    output = []
    for first, second in comparisons:
        common = sorted(
            seed for run_id, seed in lookup
            if run_id == first and (second, seed) in lookup)
        deltas = [lookup[(first, seed)]["test_weighted_f1"]
                  - lookup[(second, seed)]["test_weighted_f1"]
                  for seed in common]
        if deltas:
            mean, std = mean_std(deltas)
            output.append({"comparison": first + " - " + second,
                           "paired_seeds": len(deltas),
                           "delta_mean": mean, "delta_std": std,
                           "wins": sum(delta > 0 for delta in deltas)})
    return output


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return "-" if value != value else "{:.2f}".format(value)


def write_markdown(path, aggregates, deltas):
    lines = [
        "# Geometry ablation summary", "",
        "| ID | Runs | Test weighted F1 | Test macro F1 | Accuracy | Wheel T/A/V |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        wheel = "/".join(fmt(row[key + "_mean"]) for key in (
            "wheel_t_accuracy", "wheel_a_accuracy", "wheel_v_accuracy"))
        lines.append("| {id} | {runs} | {wf1} ± {wf1s} | {mf1} ± {mf1s} | "
                     "{acc} ± {accs} | {wheel} |".format(
                         id=row["id"], runs=row["runs"],
                         wf1=fmt(row["test_weighted_f1_mean"]),
                         wf1s=fmt(row["test_weighted_f1_std"]),
                         mf1=fmt(row["test_macro_f1_mean"]),
                         mf1s=fmt(row["test_macro_f1_std"]),
                         acc=fmt(row["test_accuracy_mean"]),
                         accs=fmt(row["test_accuracy_std"]), wheel=wheel))
    lines.extend(["", "## Paired seed deltas", "",
                  "| Comparison | Seeds | Mean delta F1 | Std | Wins |",
                  "|---|---:|---:|---:|---:|"])
    for row in deltas:
        lines.append("| {comparison} | {paired_seeds} | {delta_mean:.2f} | "
                     "{delta_std:.2f} | {wins}/{paired_seeds} |".format(**row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args()
    root = args.results_dir.resolve()
    rows = load_runs(root, args.include_smoke)
    if not rows:
        raise SystemExit("No completed geometry runs found under {}".format(root))
    duplicates = [key for key, count in Counter(
        (row["id"], row["seed"]) for row in rows).items() if count > 1]
    if duplicates:
        raise SystemExit(
            "Duplicate (ID, seed) runs found: {}. Use a clean OUTPUT_DIR "
            "for each matrix/temperature.".format(duplicates))
    aggregates = aggregate(rows)
    deltas = paired_deltas(rows)
    write_csv(root / "geometry_runs.csv", rows)
    write_csv(root / "geometry_summary.csv", aggregates)
    write_csv(root / "geometry_paired_deltas.csv", deltas)
    write_markdown(root / "geometry_summary.md", aggregates, deltas)
    print("Read {} runs; wrote summary files to {}".format(len(rows), root))


if __name__ == "__main__":
    main()
