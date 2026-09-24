"""Summarize radial, residual, and metric causal geometry ablations."""

import argparse
import csv
import json
from pathlib import Path


ORDER = ("P-FREE", "P-FIXED", "P-2D", "P-ZERORES", "E-FREE")


def causal_id(config):
    geometry = config.get("wheel_geometry", "poincare")
    dimension = int(config.get("hyperbolic_dim", 0))
    radius_mode = config.get("wheel_radius_mode", "free")
    zero_residual = bool(config.get("wheel_zero_residual", False))
    if geometry == "euclidean" and dimension == 16 and radius_mode == "free":
        return "E-FREE"
    if geometry != "poincare":
        return None
    if radius_mode == "fixed" and dimension == 16 and not zero_residual:
        return "P-FIXED"
    if radius_mode == "free" and dimension == 2 and not zero_residual:
        return "P-2D"
    if radius_mode == "free" and dimension == 16 and zero_residual:
        return "P-ZERORES"
    if radius_mode == "free" and dimension == 16 and not zero_residual:
        return "P-FREE"
    return None


def load_rows(root):
    rows = []
    for summary_path in sorted(root.rglob("summary.json")):
        config_path = summary_path.parent / "config.json"
        if not config_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if summary.get("smoke_test", False):
            continue
        run_id = causal_id(config["model_config"])
        if run_id is None:
            continue
        test = summary["test"]
        rows.append({
            "id": run_id,
            "seed": int(config["args"]["seed"]),
            "best_epoch": int(summary["best_epoch"]),
            "weighted_f1": float(test["weighted_f1"]),
            "macro_f1": float(test.get("macro_f1", float("nan"))),
            "accuracy": float(test["accuracy"]),
            "run_dir": str(summary_path.parent.resolve()),
        })
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_deltas(rows):
    lookup = {(row["id"], row["seed"]): row for row in rows}
    comparisons = (
        ("radial", "P-FREE", "P-FIXED"),
        ("dimension", "P-FREE", "P-2D"),
        ("residual", "P-FREE", "P-ZERORES"),
        ("hyperbolic_metric", "P-FREE", "E-FREE"),
    )
    output = []
    for hypothesis, first, second in comparisons:
        seeds = sorted(seed for run_id, seed in lookup
                       if run_id == first and (second, seed) in lookup)
        for seed in seeds:
            first_row, second_row = lookup[(first, seed)], lookup[(second, seed)]
            output.append({
                "hypothesis": hypothesis,
                "comparison": first + " - " + second,
                "seed": seed,
                "delta_weighted_f1": (
                    first_row["weighted_f1"] - second_row["weighted_f1"]),
                "delta_macro_f1": first_row["macro_f1"] - second_row["macro_f1"],
                "delta_accuracy": first_row["accuracy"] - second_row["accuracy"],
                "supports_hypothesis": int(
                    first_row["weighted_f1"] > second_row["weighted_f1"]),
            })
    return output


def write_markdown(path, rows, deltas):
    by_id = {row["id"]: row for row in rows}
    lines = [
        "# Causal geometry ablation", "",
        "| ID | Weighted F1 | Macro F1 | Accuracy | Best epoch |",
        "|---|---:|---:|---:|---:|",
    ]
    for run_id in ORDER:
        if run_id not in by_id:
            continue
        row = by_id[run_id]
        lines.append("| {id} | {weighted_f1:.3f} | {macro_f1:.3f} | "
                     "{accuracy:.3f} | {best_epoch} |".format(**row))
    lines.extend(["", "## Paired causal contrasts", "",
                  "| Hypothesis | Comparison | Seed | Delta weighted F1 | Supports |",
                  "|---|---|---:|---:|---:|"])
    for row in deltas:
        lines.append("| {hypothesis} | {comparison} | {seed} | "
                     "{delta_weighted_f1:+.3f} | {supports_hypothesis} |".format(
                         **row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    root = args.results_dir.expanduser().resolve()
    rows = load_rows(root)
    if not rows:
        raise SystemExit("No complete causal geometry runs found under {}".format(root))
    duplicates = {(row["id"], row["seed"]) for row in rows
                  if sum(item["id"] == row["id"] and item["seed"] == row["seed"]
                         for item in rows) > 1}
    if duplicates:
        raise SystemExit("Duplicate (ID, seed) runs: {}".format(sorted(duplicates)))
    rows.sort(key=lambda row: (ORDER.index(row["id"]), row["seed"]))
    deltas = paired_deltas(rows)
    write_csv(root / "causal_runs.csv", rows)
    write_csv(root / "causal_deltas.csv", deltas)
    write_markdown(root / "causal_summary.md", rows, deltas)
    print("Wrote causal geometry summary to {}".format(root))


if __name__ == "__main__":
    main()
