from pathlib import Path
from typing import Dict, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity

from .data import load_drug_profiles
from .utils import save_json


def masked_regression_metrics(
    true: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> Dict[str, float]:
    valid_true = true[mask]
    valid_prediction = prediction[mask]
    if valid_true.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    error = valid_prediction - valid_true
    denominator = np.sum((valid_true - valid_true.mean()) ** 2)
    r2 = (
        1.0 - np.sum(error**2) / denominator
        if denominator > 0
        else float("nan")
    )
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(r2),
    }


def per_output_metrics(
    true: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    names: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for index, name in enumerate(names):
        valid = mask[:, index] & np.isfinite(true[:, index])
        y = true[valid, index]
        p = prediction[valid, index]
        pearson = float("nan")
        spearman = float("nan")
        if len(y) >= 3 and np.ptp(y) > 1e-8 and np.ptp(p) > 1e-8:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pearson = float(pearsonr(y, p)[0])
                spearman = float(spearmanr(y, p).correlation)
        rows.append(
            {
                "name": name,
                "n": int(valid.sum()),
                "mae": float(np.mean(np.abs(p - y))) if len(y) else float("nan"),
                "rmse": float(np.sqrt(np.mean((p - y) ** 2)))
                if len(y)
                else float("nan"),
                "pearson": pearson,
                "spearman": spearman,
            }
        )
    return pd.DataFrame(rows)


def _safe_cosine(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or np.linalg.norm(left) == 0 or np.linalg.norm(right) == 0:
        return float("nan")
    return float(cosine_similarity(left.reshape(1, -1), right.reshape(1, -1))[0, 0])


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if (
        left.size < 3
        or np.ptp(left) <= 1e-8
        or np.ptp(right) <= 1e-8
    ):
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(spearmanr(left, right).correlation)


def export_drug_similarity(
    output_dir: Path,
    root: Path,
    participant_ids: Sequence[str],
    patient_common_pathway: np.ndarray,
    target_like: np.ndarray,
    target_membership: np.ndarray,
    target_names: Sequence[str],
    common_target_pathway_indices: np.ndarray,
    common_pathways: Sequence[str],
    top_k: int,
) -> None:
    drug = load_drug_profiles(root)
    if list(target_names) != drug["targets"]:
        raise ValueError("Target ordering differs from the trained model schema.")
    affinity_values = np.nan_to_num(drug["affinity"], nan=0.0)
    numerator = affinity_values.dot(target_membership)
    denominator = drug["affinity_mask"].astype(np.float32).dot(target_membership)
    drug_pathway = numerator / np.maximum(denominator, 1.0)
    drug_pathway_mask = denominator > 0
    rows = []
    for patient_index, participant_id in enumerate(participant_ids):
        for drug_index, drug_name in enumerate(drug["drugs"]):
            target_mask = drug["affinity_mask"][drug_index]
            common_path_mask = drug_pathway_mask[
                drug_index, common_target_pathway_indices
            ]
            patient_path = patient_common_pathway[patient_index, common_path_mask]
            reference_path = drug_pathway[
                drug_index, common_target_pathway_indices
            ][common_path_mask]
            patient_target = target_like[patient_index, target_mask]
            reference_target = drug["affinity"][drug_index, target_mask]
            path_cosine = _safe_cosine(patient_path, reference_path)
            path_spearman = _safe_spearman(patient_path, reference_path)
            target_cosine = _safe_cosine(patient_target, reference_target)
            target_spearman = _safe_spearman(patient_target, reference_target)
            components = np.asarray(
                [path_cosine, path_spearman, target_cosine, target_spearman],
                dtype=float,
            )
            combined = (
                float(np.nanmean(components))
                if np.isfinite(components).any()
                else float("nan")
            )
            observed_targets = np.where(target_mask)[0]
            ordered_targets = observed_targets[
                np.argsort(-drug["affinity"][drug_index, observed_targets])
            ][:5]
            pathway_values = drug_pathway[
                drug_index, common_target_pathway_indices
            ]
            valid_path_indices = np.where(common_path_mask)[0]
            ordered_paths = valid_path_indices[
                np.argsort(-pathway_values[valid_path_indices])
            ][:5]
            rows.append(
                {
                    "Participant ID": str(participant_id),
                    "drug_name": drug_name,
                    "combined_similarity": combined,
                    "pathway_cosine": path_cosine,
                    "pathway_spearman": path_spearman,
                    "target_cosine": target_cosine,
                    "target_spearman": target_spearman,
                    "major_targets": "|".join(
                        target_names[index] for index in ordered_targets
                    ),
                    "major_pathways": "|".join(
                        common_pathways[index] for index in ordered_paths
                    ),
                    "interpretation": (
                        "Pattern similarity only; not evidence of drug exposure, "
                        "binding, treatment response, or causal effect."
                    ),
                }
            )
    all_scores = pd.DataFrame(rows)
    all_scores["rank"] = all_scores.groupby("Participant ID")[
        "combined_similarity"
    ].rank(method="first", ascending=False)
    all_scores = all_scores.sort_values(["Participant ID", "rank"])
    all_scores.to_csv(
        output_dir / "drug_similarity_all.csv", index=False, encoding="utf-8-sig"
    )
    all_scores[all_scores["rank"] <= top_k].to_csv(
        output_dir / ("drug_similarity_top%d.csv" % top_k),
        index=False,
        encoding="utf-8-sig",
    )


def export_evaluation(
    output_dir: Path,
    participant_ids: Sequence[str],
    protein_names: Sequence[str],
    pathway_names: Sequence[str],
    true_delta: np.ndarray,
    predicted_delta: np.ndarray,
    protein_mask: np.ndarray,
    true_pathway: np.ndarray,
    predicted_pathway_for_metric: np.ndarray,
    pathway_mask: np.ndarray,
) -> None:
    metrics = {
        "protein": masked_regression_metrics(
            true_delta, predicted_delta, protein_mask
        ),
        "pathway": masked_regression_metrics(
            true_pathway, predicted_pathway_for_metric, pathway_mask
        ),
        "scientific_scope": (
            "Drug outputs quantify similarity to known drug-target-pathway patterns. "
            "They do not infer actual drug use, in-vivo binding, or causal drug effects."
        ),
    }
    save_json(metrics, output_dir / "metrics_summary.json")
    per_output_metrics(
        true_delta, predicted_delta, protein_mask, protein_names
    ).to_csv(
        output_dir / "metrics_per_protein.csv", index=False, encoding="utf-8-sig"
    )
    per_output_metrics(
        true_pathway, predicted_pathway_for_metric, pathway_mask, pathway_names
    ).to_csv(
        output_dir / "metrics_per_pathway.csv", index=False, encoding="utf-8-sig"
    )
