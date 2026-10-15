"""Summarize prototype-radius ablation runs without loading checkpoints."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def mean_std(values):
    if not values:
        return None, None
    return statistics.fmean(values), (
        statistics.pstdev(values) if len(values) > 1 else 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()

    rows = []
    for summary_path in sorted(args.result_dir.rglob("summary.json")):
        config_path = summary_path.with_name("config.json")
        if not config_path.is_file():
            continue
        summary = read_json(summary_path)
        config = read_json(config_path)
        run_args = config.get("args", {})
        model_config = config.get("model_config", {})
        mode = model_config.get(
            "wheel_prototype_radius_mode",
            run_args.get("wheel_prototype_radius_mode", "fixed-shared"),
        )
        if not model_config.get("use_emotion_wheel", False):
            continue
        radii = summary.get("learned_prototype_radii") or []
        radius_mean, radius_std = mean_std(radii)
        test = summary.get("test", {})
        rows.append({
            "mode": mode,
            "dataset": run_args.get("dataset"),
            "seed": run_args.get("seed"),
            "best_epoch": summary.get("best_epoch"),
            "selection_protocol": summary.get("selection_protocol"),
            "selection_weighted_f1": summary.get("selection_weighted_f1"),
            "test_accuracy": test.get("accuracy"),
            "test_weighted_f1": test.get("weighted_f1"),
            "test_macro_f1": test.get("macro_f1"),
            "radius_mean": radius_mean,
            "radius_std": radius_std,
            "radius_min": min(radii) if radii else None,
            "radius_max": max(radii) if radii else None,
            "learned_radii": ";".join("{:.8f}".format(x) for x in radii),
            "lambda_radius": run_args.get("lambda_wheel_radius", 0.0),
            "run_dir": str(summary_path.parent.resolve()),
        })

    if not rows:
        raise SystemExit("No Wheel summary.json files found under {}".format(
            args.result_dir))

    output = args.result_dir / "radius_ablation_runs.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows = []
    for mode in sorted({row["mode"] for row in rows}):
        selected = [row for row in rows if row["mode"] == mode]
        weighted_mean, weighted_std = mean_std([
            float(row["test_weighted_f1"]) for row in selected])
        macro_mean, macro_std = mean_std([
            float(row["test_macro_f1"]) for row in selected])
        aggregate_rows.append({
            "mode": mode,
            "runs": len(selected),
            "test_weighted_f1_mean": weighted_mean,
            "test_weighted_f1_std": weighted_std,
            "test_macro_f1_mean": macro_mean,
            "test_macro_f1_std": macro_std,
        })
    aggregate = args.result_dir / "radius_ablation_aggregate.csv"
    with aggregate.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    print("Read {} runs; wrote {} and {}".format(
        len(rows), output, aggregate))


if __name__ == "__main__":
    main()
