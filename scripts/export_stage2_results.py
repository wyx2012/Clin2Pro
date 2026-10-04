import argparse
import sys
import warnings
from pathlib import Path

# Support direct execution from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from clin2pro.config import load_config
from clin2pro.data import (
    Standardizer,
    TemporalDataset,
    extract_paired_features,
    load_drug_profiles,
    load_pathway_matrices,
    mimic_path,
    pair_mimic_rows,
)
from clin2pro.models import (
    ClinicalEncoder,
    DrugPretrainingModel,
    TemporalPerturbationEncoder,
)
from clin2pro.utils import resolve_device, save_json, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export Stage 2 validation predictions and metrics from an existing "
            "drug-exposure-associated perturbation checkpoint."
        )
    )
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Defaults to outputs/full/stage2/best.pt.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to outputs/full/stage2/validation_results.",
    )
    return parser.parse_args()


def safe_correlation(left, right, method):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if len(left) < 3 or np.ptp(left) <= 1e-8 or np.ptp(right) <= 1e-8:
        return float("nan")
    if method == "spearman":
        left = pd.Series(left).rank(method="average").to_numpy(dtype=float)
        right = pd.Series(right).rank(method="average").to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.corrcoef(left, right)[0, 1])


def safe_cosine(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if len(left) == 0 or denominator <= 1e-12:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def mean_absolute_error(true, prediction):
    true = np.asarray(true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    return float(np.mean(np.abs(prediction - true)))


def mean_squared_error(true, prediction):
    true = np.asarray(true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    return float(np.mean((prediction - true) ** 2))


def r2_score(true, prediction):
    true = np.asarray(true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    denominator = np.sum((true - true.mean()) ** 2)
    if denominator <= 1e-12:
        return float("nan")
    return float(1.0 - np.sum((prediction - true) ** 2) / denominator)


def accuracy_score(true, prediction):
    true = np.asarray(true, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    return float(np.mean(true == prediction))


def f1_score(true, prediction):
    true = np.asarray(true, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    true_positive = np.sum((true == 1) & (prediction == 1))
    false_positive = np.sum((true == 0) & (prediction == 1))
    false_negative = np.sum((true == 1) & (prediction == 0))
    denominator = 2 * true_positive + false_positive + false_negative
    return float(2 * true_positive / denominator) if denominator else 0.0


def binary_curve_points(true, score):
    true = np.asarray(true, dtype=int)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    sorted_true = true[order]
    sorted_score = score[order]
    distinct = np.where(np.diff(sorted_score))[0]
    threshold_indices = np.r_[distinct, len(sorted_true) - 1]
    true_positive = np.cumsum(sorted_true)[threshold_indices].astype(float)
    false_positive = (
        1 + threshold_indices - true_positive
    ).astype(float)
    positives = float(np.sum(true == 1))
    negatives = float(np.sum(true == 0))
    tpr = true_positive / positives if positives else np.full_like(true_positive, np.nan)
    fpr = (
        false_positive / negatives
        if negatives
        else np.full_like(false_positive, np.nan)
    )
    precision = true_positive / np.maximum(true_positive + false_positive, 1.0)
    recall = tpr.copy()
    thresholds = sorted_score[threshold_indices]
    roc_fpr = np.r_[0.0, fpr, 1.0]
    roc_tpr = np.r_[0.0, tpr, 1.0]
    roc_thresholds = np.r_[np.inf, thresholds, -np.inf]
    pr_precision = np.r_[1.0, precision]
    pr_recall = np.r_[0.0, recall]
    pr_thresholds = np.r_[np.nan, thresholds]
    return (
        roc_fpr,
        roc_tpr,
        roc_thresholds,
        pr_precision,
        pr_recall,
        pr_thresholds,
    )


def roc_auc_score(true, score):
    if len(np.unique(true)) != 2:
        return float("nan")
    fpr, tpr, _, _, _, _ = binary_curve_points(true, score)
    return float(np.trapz(tpr, fpr))


def average_precision_score(true, score):
    if len(np.unique(true)) != 2:
        return float("nan")
    _, _, _, precision, recall, _ = binary_curve_points(true, score)
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def pca_two_components(values):
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    left, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    components = left[:, :2] * singular_values[:2]
    total_variance = np.sum(singular_values**2)
    explained = (
        singular_values[:2] ** 2 / total_variance
        if total_variance > 0
        else np.zeros(2, dtype=float)
    )
    if components.shape[1] < 2:
        components = np.pad(
            components, ((0, 0), (0, 2 - components.shape[1]))
        )
        explained = np.pad(explained, (0, 2 - len(explained)))
    return components.astype(np.float32), explained.astype(float)


def drug_pathway_profiles(affinity, affinity_mask, membership):
    values = np.nan_to_num(affinity, nan=0.0)
    numerator = values.dot(membership)
    denominator = affinity_mask.astype(np.float32).dot(membership)
    mask = denominator > 0
    profile = numerator / np.maximum(denominator, 1.0)
    return profile.astype(np.float32), mask


def group_validation_indices(groups, test_fraction, seed):
    groups = np.asarray(groups, dtype=object)
    # Match sklearn GroupShuffleSplit: sorted unique groups, seeded permutation,
    # and ceil(test_fraction * n_groups) groups assigned to validation.
    unique_groups = np.unique(groups)
    random = np.random.RandomState(seed)
    shuffled = unique_groups.copy()
    random.shuffle(shuffled)
    n_validation = max(1, int(np.ceil(len(shuffled) * test_fraction)))
    validation_groups = set(shuffled[:n_validation].tolist())
    return np.asarray(
        [index for index, group in enumerate(groups) if group in validation_groups],
        dtype=np.int64,
    )


def build_model(config, checkpoint, device):
    model_cfg = checkpoint.get("model_config", config["model"])
    clinical = ClinicalEncoder(
        n_features=len(checkpoint["feature_names"]),
        hidden=model_cfg["clinical_hidden"],
        layers=model_cfg["clinical_layers"],
        heads=model_cfg["clinical_heads"],
        dropout=model_cfg["clinical_dropout"],
    )
    temporal = TemporalPerturbationEncoder(
        clinical_hidden=model_cfg["clinical_hidden"],
        n_features=len(checkpoint["feature_names"]),
        hidden=model_cfg["temporal_hidden"],
        dropout=model_cfg["temporal_dropout"],
    )
    model = DrugPretrainingModel(
        clinical,
        temporal,
        model_cfg["temporal_hidden"],
        len(checkpoint["target_names"]),
        len(checkpoint["target_pathways"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


def load_validation_data(config, checkpoint):
    root = Path(config["data"]["root"])
    feature_names = checkpoint["feature_names"]
    drug = load_drug_profiles(root)
    if drug["targets"] != checkpoint["target_names"]:
        raise ValueError("Target ordering differs from the Stage 2 checkpoint.")
    if drug["drugs"] != checkpoint["drug_names"]:
        raise ValueError("Drug ordering differs from the Stage 2 checkpoint.")
    pathways = load_pathway_matrices(
        root, checkpoint["protein_names"], checkpoint["target_names"]
    )
    if pathways["target_pathways"] != checkpoint["target_pathways"]:
        raise ValueError("Target pathway ordering differs from the checkpoint.")
    drug_pathway, drug_pathway_mask = drug_pathway_profiles(
        drug["affinity"], drug["affinity_mask"], pathways["target_raw"]
    )
    drug_index = {name: index for index, name in enumerate(drug["drugs"])}
    frames = []
    for imputation in config["data"]["stage2_imputations"]:
        frame = pair_mimic_rows(
            mimic_path(root, imputation),
            config["data"]["id_column"],
            config["data"].get("max_stage2_pairs_per_imputation"),
        )
        frame["_imputation"] = imputation
        frames.append(frame)
    paired = pd.concat(frames, ignore_index=True)
    x_pre, x_post = extract_paired_features(paired, feature_names)
    scaler = Standardizer.from_state_dict(checkpoint["clinical_scaler"])
    x_pre = scaler.transform(x_pre)
    x_post = scaler.transform(x_post)
    interval = paired["_interval_years"].to_numpy(dtype=np.float32)
    label_indices = np.asarray(
        [drug_index[name] for name in paired["drug_name"]], dtype=np.int64
    )
    payload = {
        "affinity": np.nan_to_num(
            drug["affinity"][label_indices], nan=0.0
        ).astype(np.float32),
        "affinity_mask": drug["affinity_mask"][label_indices],
        "binary": np.nan_to_num(
            drug["binary"][label_indices], nan=0.0
        ).astype(np.float32),
        "binary_mask": drug["binary_mask"][label_indices],
        "pathway": drug_pathway[label_indices],
        "pathway_mask": drug_pathway_mask[label_indices],
    }
    validation_indices = group_validation_indices(
        paired["_sample_key"].to_numpy(),
        config["stage2"]["val_fraction"],
        config["seed"],
    )
    validation_payload = {
        key: value[validation_indices] for key, value in payload.items()
    }
    dataset = TemporalDataset(
        x_pre[validation_indices],
        x_post[validation_indices],
        interval[validation_indices],
        validation_payload,
    )
    metadata = paired.iloc[validation_indices][
        [
            config["data"]["id_column"],
            "drug_name",
            "_episode",
            "_sample_key",
            "_imputation",
            "_interval_years",
        ]
    ].reset_index(drop=True)
    return dataset, metadata, drug, pathways


def predict(model, dataset, batch_size, num_workers, device):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    output = {
        "affinity_prediction": [],
        "binary_probability": [],
        "pathway_prediction": [],
        "temporal_embedding": [],
    }
    with torch.no_grad():
        for batch in loader:
            prediction = model(
                batch["x_pre"].to(device),
                batch["x_post"].to(device),
                batch["interval"].to(device),
            )
            output["affinity_prediction"].append(
                prediction["affinity"].cpu().numpy()
            )
            output["binary_probability"].append(
                torch.sigmoid(prediction["binary_logits"]).cpu().numpy()
            )
            output["pathway_prediction"].append(
                prediction["pathway"].cpu().numpy()
            )
            output["temporal_embedding"].append(
                prediction["temporal"].cpu().numpy()
            )
    return {key: np.concatenate(value) for key, value in output.items()}


def aggregate_imputations(metadata, dataset, predictions):
    keys = metadata["_sample_key"].drop_duplicates().tolist()
    grouped = metadata.groupby("_sample_key", sort=False).indices
    rows = []
    affinity_prediction = []
    binary_probability = []
    pathway_prediction = []
    temporal_embedding = []
    affinity_true = []
    affinity_mask = []
    binary_true = []
    binary_mask = []
    pathway_true = []
    pathway_mask = []
    for key in keys:
        indices = np.asarray(grouped[key], dtype=np.int64)
        first = int(indices[0])
        row = metadata.iloc[first].to_dict()
        row["n_imputations"] = int(len(indices))
        row["imputations"] = "|".join(metadata.iloc[indices]["_imputation"])
        rows.append(row)
        affinity_prediction.append(
            predictions["affinity_prediction"][indices].mean(axis=0)
        )
        binary_probability.append(
            predictions["binary_probability"][indices].mean(axis=0)
        )
        pathway_prediction.append(
            predictions["pathway_prediction"][indices].mean(axis=0)
        )
        temporal_embedding.append(
            predictions["temporal_embedding"][indices].mean(axis=0)
        )
        affinity_true.append(dataset.payload["affinity"][first].numpy())
        affinity_mask.append(dataset.payload["affinity_mask"][first].numpy())
        binary_true.append(dataset.payload["binary"][first].numpy())
        binary_mask.append(dataset.payload["binary_mask"][first].numpy())
        pathway_true.append(dataset.payload["pathway"][first].numpy())
        pathway_mask.append(dataset.payload["pathway_mask"][first].numpy())
    return {
        "metadata": pd.DataFrame(rows),
        "affinity_prediction": np.stack(affinity_prediction),
        "binary_probability": np.stack(binary_probability),
        "pathway_prediction": np.stack(pathway_prediction),
        "temporal_embedding": np.stack(temporal_embedding),
        "affinity_true": np.stack(affinity_true),
        "affinity_mask": np.stack(affinity_mask).astype(bool),
        "binary_true": np.stack(binary_true),
        "binary_mask": np.stack(binary_mask).astype(bool),
        "pathway_true": np.stack(pathway_true),
        "pathway_mask": np.stack(pathway_mask).astype(bool),
    }


def write_wide(path, metadata, values, columns):
    frame = pd.DataFrame(values, columns=columns)
    for column in reversed(
        ["Participant ID", "drug_name", "_episode", "_sample_key"]
    ):
        if column in metadata:
            frame.insert(0, column, metadata[column].to_numpy())
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def profile_metrics(true, prediction, mask):
    rows = []
    for index in range(len(true)):
        valid = mask[index]
        y = true[index, valid]
        p = prediction[index, valid]
        rows.append(
            {
                "n_observed": int(valid.sum()),
                "mae": float(mean_absolute_error(y, p)) if len(y) else np.nan,
                "rmse": float(np.sqrt(mean_squared_error(y, p)))
                if len(y)
                else np.nan,
                "pearson": safe_correlation(y, p, "pearson"),
                "spearman": safe_correlation(y, p, "spearman"),
                "cosine": safe_cosine(y, p),
            }
        )
    return pd.DataFrame(rows)


def per_dimension_regression_metrics(true, prediction, mask, names, name_column):
    rows = []
    for index, name in enumerate(names):
        valid = mask[:, index]
        y = true[valid, index]
        p = prediction[valid, index]
        rows.append(
            {
                name_column: name,
                "n_observed": int(valid.sum()),
                "mae": float(mean_absolute_error(y, p)) if len(y) else np.nan,
                "rmse": float(np.sqrt(mean_squared_error(y, p)))
                if len(y)
                else np.nan,
                "r2_score": float(r2_score(y, p))
                if len(y) >= 2 and np.ptp(y) > 1e-8
                else np.nan,
                "pearson": safe_correlation(y, p, "pearson"),
                "spearman": safe_correlation(y, p, "spearman"),
            }
        )
    return pd.DataFrame(rows)


def binary_metrics(true, probability, mask, target_names):
    y = true[mask].astype(int)
    score = probability[mask]
    prediction = (score >= 0.5).astype(int)
    summary = {
        "n_observed_labels": int(len(y)),
        "positive_labels": int(y.sum()),
        "positive_fraction": float(y.mean()),
        "auroc": float(roc_auc_score(y, score))
        if len(np.unique(y)) == 2
        else np.nan,
        "auprc": float(average_precision_score(y, score))
        if len(np.unique(y)) == 2
        else np.nan,
        "f1_at_0_5": float(f1_score(y, prediction)),
        "accuracy_at_0_5": float(accuracy_score(y, prediction)),
    }
    rows = []
    for index, target in enumerate(target_names):
        valid = mask[:, index]
        target_y = true[valid, index].astype(int)
        target_score = probability[valid, index]
        target_prediction = (target_score >= 0.5).astype(int)
        rows.append(
            {
                "target": target,
                "n_observed": int(valid.sum()),
                "n_positive": int(target_y.sum()),
                "auroc": float(roc_auc_score(target_y, target_score))
                if len(np.unique(target_y)) == 2
                else np.nan,
                "auprc": float(
                    average_precision_score(target_y, target_score)
                )
                if len(np.unique(target_y)) == 2
                else np.nan,
                "f1_at_0_5": float(f1_score(target_y, target_prediction))
                if len(target_y)
                else np.nan,
            }
        )
    (
        fpr,
        tpr,
        roc_threshold,
        precision,
        recall,
        pr_threshold,
    ) = binary_curve_points(y, score)
    roc_points = pd.DataFrame(
        {
            "false_positive_rate": fpr,
            "true_positive_rate": tpr,
            "threshold": roc_threshold,
        }
    )
    pr_points = pd.DataFrame(
        {
            "recall": recall,
            "precision": precision,
            "threshold": pr_threshold,
        }
    )
    return summary, pd.DataFrame(rows), roc_points, pr_points


def aggregate_by_drug(result, target_names, pathway_names):
    metadata = result["metadata"]
    rows = []
    affinity_predicted = []
    affinity_true = []
    pathway_predicted = []
    pathway_true = []
    for drug_name, indices in metadata.groupby("drug_name", sort=True).indices.items():
        indices = np.asarray(indices, dtype=np.int64)
        rows.append(
            {
                "drug_name": drug_name,
                "n_validation_episodes": int(len(indices)),
            }
        )
        affinity_predicted.append(
            result["affinity_prediction"][indices].mean(axis=0)
        )
        affinity_true.append(
            np.where(
                result["affinity_mask"][indices[0]],
                result["affinity_true"][indices[0]],
                np.nan,
            )
        )
        pathway_predicted.append(
            result["pathway_prediction"][indices].mean(axis=0)
        )
        pathway_true.append(
            np.where(
                result["pathway_mask"][indices[0]],
                result["pathway_true"][indices[0]],
                np.nan,
            )
        )
    metadata_drug = pd.DataFrame(rows)
    return (
        metadata_drug,
        pd.DataFrame(affinity_predicted, columns=target_names),
        pd.DataFrame(affinity_true, columns=target_names),
        pd.DataFrame(pathway_predicted, columns=pathway_names),
        pd.DataFrame(pathway_true, columns=pathway_names),
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config["device"])
    logger = setup_logging(config["output_dir"])
    checkpoint_path = (
        Path(args.checkpoint).resolve()
        if args.checkpoint
        else Path(config["output_dir"]) / "stage2" / "best.pt"
    )
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(config["output_dir"]) / "stage2" / "validation_results"
    )
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    dataset, metadata, drug, pathways = load_validation_data(config, checkpoint)
    model = build_model(config, checkpoint, device)
    predictions = predict(
        model,
        dataset,
        config["stage2"]["batch_size"],
        config["num_workers"],
        device,
    )
    result = aggregate_imputations(metadata, dataset, predictions)
    result["metadata"].to_csv(
        output / "validation_episode_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_wide(
        output / "predicted_affinity_validation.csv",
        result["metadata"],
        result["affinity_prediction"],
        checkpoint["target_names"],
    )
    write_wide(
        output / "observed_affinity_validation.csv",
        result["metadata"],
        np.where(result["affinity_mask"], result["affinity_true"], np.nan),
        checkpoint["target_names"],
    )
    write_wide(
        output / "predicted_binary_probability_validation.csv",
        result["metadata"],
        result["binary_probability"],
        checkpoint["target_names"],
    )
    write_wide(
        output / "observed_binary_interaction_validation.csv",
        result["metadata"],
        np.where(result["binary_mask"], result["binary_true"], np.nan),
        checkpoint["target_names"],
    )
    write_wide(
        output / "predicted_drug_pathway_validation.csv",
        result["metadata"],
        result["pathway_prediction"],
        checkpoint["target_pathways"],
    )
    write_wide(
        output / "observed_drug_pathway_validation.csv",
        result["metadata"],
        np.where(result["pathway_mask"], result["pathway_true"], np.nan),
        checkpoint["target_pathways"],
    )

    embedding = result["temporal_embedding"]
    embedding_pca, pca_explained_variance_ratio = pca_two_components(embedding)
    embedding_frame = result["metadata"][
        ["Participant ID", "drug_name", "_episode", "_sample_key"]
    ].copy()
    embedding_frame["PCA1"] = embedding_pca[:, 0]
    embedding_frame["PCA2"] = embedding_pca[:, 1]
    embedding_frame.to_csv(
        output / "temporal_embedding_pca_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_wide(
        output / "temporal_embedding_validation.csv",
        result["metadata"],
        embedding,
        ["latent_%03d" % index for index in range(embedding.shape[1])],
    )

    affinity_profile = profile_metrics(
        result["affinity_true"],
        result["affinity_prediction"],
        result["affinity_mask"],
    )
    affinity_profile = pd.concat(
        [
            result["metadata"][
                ["Participant ID", "drug_name", "_episode", "_sample_key"]
            ].reset_index(drop=True),
            affinity_profile,
        ],
        axis=1,
    )
    affinity_profile.to_csv(
        output / "affinity_metrics_per_episode.csv",
        index=False,
        encoding="utf-8-sig",
    )
    per_dimension_regression_metrics(
        result["affinity_true"],
        result["affinity_prediction"],
        result["affinity_mask"],
        checkpoint["target_names"],
        "target",
    ).to_csv(
        output / "affinity_metrics_per_target.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pathway_profile = profile_metrics(
        result["pathway_true"],
        result["pathway_prediction"],
        result["pathway_mask"],
    )
    pathway_profile = pd.concat(
        [
            result["metadata"][
                ["Participant ID", "drug_name", "_episode", "_sample_key"]
            ].reset_index(drop=True),
            pathway_profile,
        ],
        axis=1,
    )
    pathway_profile.to_csv(
        output / "drug_pathway_metrics_per_episode.csv",
        index=False,
        encoding="utf-8-sig",
    )
    per_dimension_regression_metrics(
        result["pathway_true"],
        result["pathway_prediction"],
        result["pathway_mask"],
        checkpoint["target_pathways"],
        "pathway",
    ).to_csv(
        output / "drug_pathway_metrics_per_pathway.csv",
        index=False,
        encoding="utf-8-sig",
    )

    binary_summary, binary_target, roc_points, pr_points = binary_metrics(
        result["binary_true"],
        result["binary_probability"],
        result["binary_mask"],
        checkpoint["target_names"],
    )
    binary_target.to_csv(
        output / "binary_metrics_per_target.csv",
        index=False,
        encoding="utf-8-sig",
    )
    roc_points.to_csv(
        output / "binary_roc_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pr_points.to_csv(
        output / "binary_precision_recall_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (
        drug_metadata,
        affinity_predicted_drug,
        affinity_true_drug,
        pathway_predicted_drug,
        pathway_true_drug,
    ) = aggregate_by_drug(
        result, checkpoint["target_names"], checkpoint["target_pathways"]
    )
    for frame in (
        affinity_predicted_drug,
        affinity_true_drug,
        pathway_predicted_drug,
        pathway_true_drug,
    ):
        frame.insert(0, "drug_name", drug_metadata["drug_name"])
        frame.insert(
            1,
            "n_validation_episodes",
            drug_metadata["n_validation_episodes"],
        )
    affinity_predicted_drug.to_csv(
        output / "predicted_affinity_profile_by_drug.csv",
        index=False,
        encoding="utf-8-sig",
    )
    affinity_true_drug.to_csv(
        output / "observed_affinity_profile_by_drug.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pathway_predicted_drug.to_csv(
        output / "predicted_pathway_profile_by_drug.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pathway_true_drug.to_csv(
        output / "observed_pathway_profile_by_drug.csv",
        index=False,
        encoding="utf-8-sig",
    )

    affinity_valid = result["affinity_mask"]
    affinity_y = result["affinity_true"][affinity_valid]
    affinity_p = result["affinity_prediction"][affinity_valid]
    pathway_valid = result["pathway_mask"]
    pathway_y = result["pathway_true"][pathway_valid]
    pathway_p = result["pathway_prediction"][pathway_valid]
    summary = {
        "evaluation": "Stage 2 held-out validation set",
        "n_validation_rows_before_imputation_aggregation": int(len(metadata)),
        "n_unique_validation_episodes": int(len(result["metadata"])),
        "n_validation_drugs": int(result["metadata"]["drug_name"].nunique()),
        "n_targets": int(len(checkpoint["target_names"])),
        "n_target_pathways": int(len(checkpoint["target_pathways"])),
        "affinity": {
            "micro_mae": float(mean_absolute_error(affinity_y, affinity_p)),
            "micro_rmse": float(
                np.sqrt(mean_squared_error(affinity_y, affinity_p))
            ),
            "micro_r2_score": float(r2_score(affinity_y, affinity_p)),
            "median_episode_pearson": float(
                affinity_profile["pearson"].median(skipna=True)
            ),
            "median_episode_spearman": float(
                affinity_profile["spearman"].median(skipna=True)
            ),
            "median_episode_cosine": float(
                affinity_profile["cosine"].median(skipna=True)
            ),
        },
        "binary_interaction": binary_summary,
        "drug_pathway": {
            "micro_mae": float(mean_absolute_error(pathway_y, pathway_p)),
            "micro_rmse": float(
                np.sqrt(mean_squared_error(pathway_y, pathway_p))
            ),
            "micro_r2_score": float(r2_score(pathway_y, pathway_p)),
            "median_episode_pearson": float(
                pathway_profile["pearson"].median(skipna=True)
            ),
            "median_episode_spearman": float(
                pathway_profile["spearman"].median(skipna=True)
            ),
            "median_episode_cosine": float(
                pathway_profile["cosine"].median(skipna=True)
            ),
        },
        "pca_explained_variance_ratio": pca_explained_variance_ratio.tolist(),
        "interpretation": (
            "These are drug-exposure-associated clinical perturbation validation "
            "results. They do not estimate pure causal drug effects."
        ),
        "validation_scope": (
            "The split holds out clinical exposure episodes, not entire drug "
            "identities. The same known drug may occur in both training and "
            "validation sets. The validation set was also used for early stopping, "
            "so these results are suitable for pretraining characterization but "
            "are not an independent external test."
        ),
    }
    save_json(summary, output / "stage2_validation_metrics_summary.json")
    logger.info(
        "Stage 2 validation exports completed: %d unique episodes -> %s",
        len(result["metadata"]),
        output,
    )


if __name__ == "__main__":
    main()
