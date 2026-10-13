"""Fit adjusted mixed-effects logistic models for structural contextual harm.

The model is an offline explanatory analysis.  Ground-truth emotion is allowed
as a control variable here and must not be confused with the deployable harm
predictor in ``train_harm_predictor.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from experiment_analysis_common import STRATEGIES, resolve_impact_paths, write_rows


FORMULA = (
    "harm ~ same_speaker + log_distance_z + emotion_difference + baseline_margin_z "
    "+ C(modality, Treatment(reference='text')) + source_before_target "
    "+ correct_full + C(target_emotion) "
    "+ same_speaker:emotion_difference "
    "+ C(modality, Treatment(reference='text')):log_distance_z"
)
RANDOM_EFFECTS = {
    "conversation_random_intercept": "0 + C(conversation_id)",
    "target_random_intercept": "0 + C(target_key)",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--impact-eps-values", type=float, nargs="+",
        default=(1e-6, 1e-4, 1e-3),
    )
    parser.add_argument(
        "--strategies", nargs="+", choices=STRATEGIES, default=STRATEGIES
    )
    parser.add_argument(
        "--max-rows", type=int, default=0,
        help="Deterministic diagnostic subsample; 0 uses every row",
    )
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args(argv)


def _baseline_margins(run_dir: Path):
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("Mixed-effects analysis requires pandas") from error

    path = Path(run_dir) / "baseline" / "baseline_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        probability_columns = sorted(
            (name for name in fieldnames if name.startswith("prob_") and name != "prob_true"),
            key=lambda name: int(name.split("_")[1]),
        )
        required = {"conversation_id", "utterance_index"}
        missing = sorted(required - set(fieldnames))
        if missing or len(probability_columns) < 2:
            raise ValueError(
                f"Invalid baseline file: missing={missing}, probabilities={probability_columns}"
            )
        rows = []
        for row in reader:
            probabilities = sorted(float(row[name]) for name in probability_columns)
            rows.append({
                "conversation_id": row["conversation_id"],
                "target_idx": int(row["utterance_index"]),
                "baseline_margin": probabilities[-1] - probabilities[-2],
            })
    frame = pd.DataFrame(rows)
    if frame.duplicated(["conversation_id", "target_idx"]).any():
        raise ValueError("Duplicate baseline target keys")
    return frame


def _load_strategy_frame(
    path: Path, baseline, max_rows: int, seed: int
):
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("Mixed-effects analysis requires pandas") from error

    columns = [
        "conversation_id", "source_idx", "target_idx", "same_speaker",
        "absolute_distance", "source_before_target", "modality",
        "replacement_strategy", "source_emotion", "target_emotion",
        "signed_impact", "correct_full",
    ]
    frame = pd.read_csv(path, usecols=columns)
    if frame.empty:
        raise ValueError(f"No rows in {path}")
    if max_rows < 0:
        raise ValueError("max_rows must be nonnegative")
    if max_rows and len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=seed).copy()
    frame = frame.merge(
        baseline,
        how="left",
        on=["conversation_id", "target_idx"],
        validate="many_to_one",
    )
    if frame["baseline_margin"].isna().any():
        raise ValueError(f"Missing baseline margins after joining {path}")
    frame["same_speaker"] = frame["same_speaker"].astype(np.int8)
    frame["source_before_target"] = frame["source_before_target"].astype(np.int8)
    frame["correct_full"] = frame["correct_full"].astype(np.int8)
    frame["emotion_difference"] = (
        frame["source_emotion"] != frame["target_emotion"]
    ).astype(np.int8)
    frame["log_distance"] = np.log1p(frame["absolute_distance"].astype(float))
    for source, target in (
        ("log_distance", "log_distance_z"),
        ("baseline_margin", "baseline_margin_z"),
    ):
        standard_deviation = float(frame[source].std(ddof=0))
        if standard_deviation == 0:
            raise ValueError(f"Cannot standardize constant covariate {source}")
        frame[target] = (
            frame[source] - float(frame[source].mean())
        ) / standard_deviation
    frame["target_key"] = (
        frame["conversation_id"].astype(str)
        + ":"
        + frame["target_idx"].astype(str)
    )
    return frame


def _normal_probability_positive(mean: float, standard_deviation: float) -> float:
    if standard_deviation <= 0:
        return float(mean > 0)
    return 0.5 * (1.0 + math.erf(mean / standard_deviation / math.sqrt(2.0)))


def fit_models(
    run_dir: Path,
    impact_eps_values: Sequence[float] = (1e-6, 1e-4, 1e-3),
    strategies: Sequence[str] = STRATEGIES,
    max_rows: int = 0,
    seed: int = 2024,
) -> Path:
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except ImportError as error:
        raise RuntimeError(
            "Mixed-effects analysis requires statsmodels>=0.14; install requirements.txt"
        ) from error

    run_dir = Path(run_dir).expanduser().resolve()
    result_dir = run_dir / "results"
    paths = {
        path.stem.removeprefix("impact_pairs_"): path
        for path in resolve_impact_paths(run_dir)
    }
    baseline = _baseline_margins(run_dir)
    fixed_rows = []
    variance_rows = []
    fit_metadata: Dict[str, object] = {
        "formula": FORMULA,
        "random_effects": RANDOM_EFFECTS,
        "impact_eps_values": list(impact_eps_values),
        "max_rows": max_rows,
        "seed": seed,
        "fits": [],
    }
    for strategy in strategies:
        if strategy not in paths:
            raise FileNotFoundError(f"Missing impact file for strategy={strategy}")
        frame = _load_strategy_frame(paths[strategy], baseline, max_rows, seed)
        for epsilon in impact_eps_values:
            if epsilon < 0:
                raise ValueError("Impact thresholds must be nonnegative")
            frame["harm"] = (frame["signed_impact"] < -epsilon).astype(np.int8)
            prevalence = float(frame["harm"].mean())
            if prevalence in (0.0, 1.0):
                raise ValueError(
                    f"Harm outcome has one class for {strategy}, epsilon={epsilon}"
                )
            print(
                f"Fitting mixed-effects logistic: strategy={strategy}, "
                f"epsilon={epsilon:g}, rows={len(frame)}, prevalence={prevalence:.4f}",
                flush=True,
            )
            model = BinomialBayesMixedGLM.from_formula(
                FORMULA, RANDOM_EFFECTS, frame
            )
            result = model.fit_vb()
            for name, mean, standard_deviation in zip(
                model.fep_names, result.fe_mean, result.fe_sd
            ):
                mean = float(mean)
                standard_deviation = float(standard_deviation)
                low = mean - 1.96 * standard_deviation
                high = mean + 1.96 * standard_deviation
                fixed_rows.append({
                    "replacement_strategy": strategy,
                    "impact_eps": epsilon,
                    "row_count": len(frame),
                    "harmful_prevalence": prevalence,
                    "term": name,
                    "log_odds_mean": mean,
                    "log_odds_sd": standard_deviation,
                    "log_odds_ci95_low": low,
                    "log_odds_ci95_high": high,
                    "odds_ratio": math.exp(mean),
                    "odds_ratio_ci95_low": math.exp(low),
                    "odds_ratio_ci95_high": math.exp(high),
                    "posterior_probability_positive": _normal_probability_positive(
                        mean, standard_deviation
                    ),
                })
            for name, log_sd_mean, log_sd_sd in zip(
                model.vcp_names, result.vcp_mean, result.vcp_sd
            ):
                variance_rows.append({
                    "replacement_strategy": strategy,
                    "impact_eps": epsilon,
                    "random_effect": name,
                    "log_standard_deviation_mean": float(log_sd_mean),
                    "log_standard_deviation_sd": float(log_sd_sd),
                    "standard_deviation": math.exp(float(log_sd_mean)),
                })
            fit_metadata["fits"].append({
                "replacement_strategy": strategy,
                "impact_eps": epsilon,
                "row_count": len(frame),
                "harmful_prevalence": prevalence,
                "optimizer": str(getattr(result, "optim_retvals", None)),
            })

    robustness_rows = []
    terms = sorted({row["term"] for row in fixed_rows})
    for term in terms:
        items = [row for row in fixed_rows if row["term"] == term]
        signs = [np.sign(float(row["log_odds_mean"])) for row in items]
        robust_positive = all(
            float(row["log_odds_ci95_low"]) > 0 for row in items
        )
        robust_negative = all(
            float(row["log_odds_ci95_high"]) < 0 for row in items
        )
        robustness_rows.append({
            "term": term,
            "fit_count": len(items),
            "positive_coefficient_count": sum(value > 0 for value in signs),
            "negative_coefficient_count": sum(value < 0 for value in signs),
            "consistent_sign_across_strategy_and_epsilon": int(
                all(value == signs[0] for value in signs)
            ),
            "ci_excludes_zero_same_direction_in_every_fit": int(
                robust_positive or robust_negative
            ),
            "robust_direction": (
                "positive" if robust_positive else "negative" if robust_negative
                else "not_robust"
            ),
            "minimum_odds_ratio": min(float(row["odds_ratio"]) for row in items),
            "maximum_odds_ratio": max(float(row["odds_ratio"]) for row in items),
        })

    write_rows(
        result_dir / "structural_harm_mixed_effects_fixed.csv", fixed_rows
    )
    write_rows(
        result_dir / "structural_harm_mixed_effects_random.csv", variance_rows
    )
    write_rows(
        result_dir / "structural_harm_mixed_effects_robustness.csv", robustness_rows
    )
    metadata_path = result_dir / "structural_harm_mixed_effects_metadata.json"
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(fit_metadata, indent=2, default=str), encoding="utf-8"
    )
    os.replace(temporary, metadata_path)
    print(f"Mixed-effects analysis complete: {result_dir}", flush=True)
    return result_dir


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    return fit_models(
        args.run_dir,
        args.impact_eps_values,
        args.strategies,
        args.max_rows,
        args.seed,
    )


if __name__ == "__main__":
    main()
