"""Summarize the fixed shared prototype-radius sweep."""

import argparse
import csv
import json
import re
from pathlib import Path


RADIUS_DIR = re.compile(r"^radius_(\d+)p(\d+)$")


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def requested_radius(result_dir, summary_path):
    relative_parts = summary_path.relative_to(result_dir).parts
    for part in relative_parts:
        match = RADIUS_DIR.match(part)
        if match:
            return float("{}.{}".format(match.group(1), match.group(2)))
    return None


def fmt(value):
    return "" if value is None else "{:.3f}".format(float(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()

    rows = []
    for summary_path in sorted(result_dir.rglob("summary.json")):
        config_path = summary_path.with_name("config.json")
        if not config_path.is_file():
            continue
        config = read_json(config_path)
        summary = read_json(summary_path)
        run_args = config.get("args", {})
        model_config = config.get("model_config", {})
        if model_config.get("wheel_prototype_radius_mode") != "fixed-shared":
            continue

        requested = requested_radius(result_dir, summary_path)
        if requested is None:
            continue
        effective = float(model_config.get(
            "wheel_prototype_radius",
            run_args.get("wheel_prototype_radius"),
        ))
        test = summary.get("test", {})
        rows.append({
            "requested_radius": requested,
            "effective_radius": effective,
            "dataset": run_args.get("dataset"),
            "seed": run_args.get("seed"),
            "selection_protocol": summary.get("selection_protocol"),
            "best_epoch": summary.get("best_epoch"),
            "test_accuracy": test.get("accuracy"),
            "test_weighted_f1": test.get("weighted_f1"),
            "test_macro_f1": test.get("macro_f1"),
            "delta_weighted_f1_vs_0p75": None,
            "run_dir": str(summary_path.parent.resolve()),
        })

    if not rows:
        raise SystemExit(
            "No fixed-shared radius sweep runs found under {}".format(
                result_dir))

    baselines = {
        (row["dataset"], row["seed"], row["selection_protocol"]):
            float(row["test_weighted_f1"])
        for row in rows
        if abs(row["requested_radius"] - 0.75) < 1e-12
    }
    for row in rows:
        key = (row["dataset"], row["seed"], row["selection_protocol"])
        if key in baselines:
            row["delta_weighted_f1_vs_0p75"] = (
                float(row["test_weighted_f1"]) - baselines[key])

    rows.sort(key=lambda row: (
        str(row["dataset"]), int(row["seed"]), row["requested_radius"]))
    csv_path = result_dir / "fixed_radius_sweep_runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    md_path = result_dir / "fixed_radius_sweep_summary.md"
    with md_path.open("w", encoding="utf-8") as stream:
        stream.write("# Fixed prototype-radius sweep\n\n")
        stream.write(
            "`requested_radius=1.0` uses the effective radius shown below "
            "because a Poincaré-ball point must have norm smaller than 1.\n\n")
        stream.write(
            "| Requested radius | Effective radius | Seed | Best epoch | "
            "Accuracy | Weighted F1 | Macro F1 | Δ WF1 vs 0.75 |\n")
        stream.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            stream.write(
                "| {requested_radius:.2f} | {effective_radius:.5f} | "
                "{seed} | {best_epoch} | {accuracy} | {weighted} | "
                "{macro} | {delta} |\n".format(
                    requested_radius=row["requested_radius"],
                    effective_radius=row["effective_radius"],
                    seed=row["seed"],
                    best_epoch=row["best_epoch"],
                    accuracy=fmt(row["test_accuracy"]),
                    weighted=fmt(row["test_weighted_f1"]),
                    macro=fmt(row["test_macro_f1"]),
                    delta=fmt(row["delta_weighted_f1_vs_0p75"]),
                ))
        stream.write("\nSource: `fixed_radius_sweep_runs.csv`.\n")

    print("Read {} runs; wrote {} and {}".format(
        len(rows), csv_path, md_path))


if __name__ == "__main__":
    main()

