"""Export leakage-safe Stage 3 baselines and detailed OOF metrics.

Run from the project root:
    python scripts/export_stage3_benchmarks.py --config configs/full.yaml

This script does not retrain the full Stage 3 model. It reuses its existing
out-of-fold predictions and trains all comparison baselines on the identical
participant-level outer folds.
"""

import argparse
import sys
import copy
import json
import math
import time
from pathlib import Path

# Support direct execution from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, KFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from clin2pro.config import load_config
from clin2pro.models import ClinicalEncoder, ProteinPredictionHead
from clin2pro.stage3 import _load_longitudinal
from clin2pro.utils import resolve_device, set_seed


MODEL_ORDER = [
    "zero_change",
    "training_mean_delta",
    "ridge",
    "elastic_net",
    "plain_mlp",
    "stage1_cross_sectional_difference",
    "full_model",
]

MODEL_LABELS = {
    "zero_change": "Zero change",
    "training_mean_delta": "Training-set mean delta",
    "ridge": "Ridge",
    "elastic_net": "ElasticNet",
    "plain_mlp": "Plain MLP",
    "stage1_cross_sectional_difference": "Stage 1: f(post)-f(pre)",
    "full_model": "Full model",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Stage 3 OOF baselines and detailed metrics."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output",
        default=None,
        help="Default: <output_dir>/stage3/benchmark_results",
    )
    parser.add_argument(
        "--full-predictions",
        default=None,
        help="Existing full-model OOF prediction CSV.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs-linear", type=int, default=350)
    parser.add_argument("--max-epochs-mlp", type=int, default=500)
    parser.add_argument("--patience", type=int, default=45)
    parser.add_argument(
        "--ridge-alphas",
        default="0.0001,0.001,0.01",
        help="Comma-separated L2 coefficients for masked linear regression.",
    )
    parser.add_argument(
        "--elasticnet-settings",
        default="0.0001:0.25,0.001:0.5,0.01:0.75",
        help="Comma-separated alpha:l1_ratio settings.",
    )
    parser.add_argument("--mlp-hidden", type=int, default=256)
    parser.add_argument("--mlp-dropout", type=float, default=0.20)
    parser.add_argument("--learning-rate-linear", type=float, default=0.003)
    parser.add_argument("--learning-rate-mlp", type=float, default=0.001)
    parser.add_argument("--min-protein-n", type=int, default=20)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Debug mode with fewer epochs and one regularization setting.",
    )
    return parser.parse_args()


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_json(payload, path):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)


def json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Not JSON serializable: %r" % (value,))


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_elastic_settings(value):
    settings = []
    for item in value.split(","):
        if not item.strip():
            continue
        alpha, ratio = item.split(":")
        settings.append(
            {"alpha": float(alpha.strip()), "l1_ratio": float(ratio.strip())}
        )
    return settings


def safe_name(value):
    return str(value).replace(" ", "_").replace("/", "_").replace("\\", "_")


def masked_mean_scale(values, mask):
    count = mask.sum(axis=0)
    total = np.where(mask, values, 0.0).sum(axis=0)
    mean = total / np.maximum(count, 1)
    centered = np.where(mask, values - mean, 0.0)
    variance = (centered * centered).sum(axis=0) / np.maximum(count, 1)
    scale = np.sqrt(variance)
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    scale = np.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0)
    scale[scale < 1e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def feature_matrix(data):
    x_pre = data["x_pre"].astype(np.float32)
    x_post = data["x_post"].astype(np.float32)
    interval = data["interval"].astype(np.float32).reshape(-1, 1)
    delta = x_post - x_pre
    rate = np.clip(delta / np.maximum(interval, 1.0 / 365.25), -20.0, 20.0)
    log_days = np.log1p(np.maximum(interval, 0.0) * 365.25)
    return np.concatenate(
        [x_pre, x_post, delta, rate, interval, log_days], axis=1
    ).astype(np.float32)


def inner_split(row_indices, groups, config):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config["stage3"]["validation_fraction"],
        random_state=config["seed"],
    )
    relative_train, relative_val = next(
        splitter.split(row_indices, groups=groups[row_indices])
    )
    return row_indices[relative_train], row_indices[relative_val]


class LinearProteomeModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


class PlainMLP(nn.Module):
    def __init__(self, input_dim, hidden, output_dim, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.network(x)


def masked_mse_torch(prediction, target, mask):
    error = (prediction - target).pow(2)
    return error.masked_select(mask).mean()


def masked_huber_torch(prediction, target, mask, delta=1.0):
    absolute = (prediction - target).abs()
    quadratic = torch.minimum(
        absolute, torch.tensor(delta, device=absolute.device)
    )
    linear = absolute - quadratic
    loss = 0.5 * quadratic.pow(2) + delta * linear
    return loss.masked_select(mask).mean()


def regularization(model, kind, alpha, l1_ratio):
    weight = model.linear.weight
    if kind == "ridge":
        return alpha * weight.pow(2).mean()
    if kind == "elastic_net":
        l1 = weight.abs().mean()
        l2 = weight.pow(2).mean()
        return alpha * (l1_ratio * l1 + 0.5 * (1.0 - l1_ratio) * l2)
    return weight.new_tensor(0.0)


def prepare_fold_arrays(features, targets, train_rows, val_rows):
    x_mean = features[train_rows].mean(axis=0)
    x_scale = features[train_rows].std(axis=0)
    x_scale[x_scale < 1e-6] = 1.0
    y_mask = np.isfinite(targets)
    y_mean, y_scale = masked_mean_scale(
        targets[train_rows], y_mask[train_rows]
    )

    def transform(rows):
        x = (features[rows] - x_mean) / x_scale
        y = (targets[rows] - y_mean) / y_scale
        mask = y_mask[rows]
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return (
            x.astype(np.float32),
            y.astype(np.float32),
            mask.astype(np.bool_),
        )

    return transform(train_rows), transform(val_rows), x_mean, x_scale, y_mean, y_scale


def fit_candidate(
    kind,
    settings,
    train_arrays,
    val_arrays,
    input_dim,
    output_dim,
    args,
    device,
    seed,
):
    set_seed(seed)
    if kind == "plain_mlp":
        model = PlainMLP(
            input_dim, args.mlp_hidden, output_dim, args.mlp_dropout
        ).to(device)
        learning_rate = args.learning_rate_mlp
        max_epochs = args.max_epochs_mlp
        weight_decay = 1e-4
    else:
        model = LinearProteomeModel(input_dim, output_dim).to(device)
        learning_rate = args.learning_rate_linear
        max_epochs = args.max_epochs_linear
        weight_decay = 0.0

    x_train, y_train, mask_train = train_arrays
    x_val, y_val, mask_val = val_arrays
    dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
        torch.from_numpy(mask_train),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )
    xv = torch.from_numpy(x_val).to(device)
    yv = torch.from_numpy(y_val).to(device)
    mv = torch.from_numpy(mask_val).to(device)
    best_loss = float("inf")
    best_state = None
    bad_epochs = 0
    epochs_run = 0
    for epoch in range(max_epochs):
        model.train()
        for xb, yb, mb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            if kind == "plain_mlp":
                loss = masked_huber_torch(prediction, yb, mb)
            else:
                loss = masked_mse_torch(prediction, yb, mb)
                loss = loss + regularization(
                    model,
                    kind,
                    settings.get("alpha", 0.0),
                    settings.get("l1_ratio", 0.0),
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_prediction = model(xv)
            val_loss = masked_mse_torch(val_prediction, yv, mv).item()
        scheduler.step(val_loss)
        epochs_run = epoch + 1
        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, best_loss, epochs_run


def fit_supervised_baseline(
    kind,
    candidate_settings,
    features,
    targets,
    train_rows,
    val_rows,
    test_rows,
    metadata,
    args,
    device,
    seed,
):
    (
        train_arrays,
        val_arrays,
        x_mean,
        x_scale,
        y_mean,
        y_scale,
    ) = prepare_fold_arrays(features, targets, train_rows, val_rows)
    records = []
    best = None
    for candidate_index, settings in enumerate(candidate_settings):
        model, val_loss, epochs_run = fit_candidate(
            kind,
            settings,
            train_arrays,
            val_arrays,
            features.shape[1],
            targets.shape[1],
            args,
            device,
            seed + candidate_index,
        )
        record = dict(settings)
        record.update(
            {
                "validation_standardized_mse": float(val_loss),
                "epochs_run": int(epochs_run),
            }
        )
        records.append(record)
        if best is None or val_loss < best["val_loss"]:
            best = {
                "model": model,
                "val_loss": val_loss,
                "settings": dict(settings),
                "epochs_run": epochs_run,
            }

    xt = ((features[test_rows] - x_mean) / x_scale).astype(np.float32)
    best["model"].eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(xt), args.batch_size):
            batch = torch.from_numpy(xt[start : start + args.batch_size]).to(device)
            predictions.append(best["model"](batch).cpu().numpy())
    row_prediction = np.concatenate(predictions, axis=0)
    row_prediction = row_prediction * y_scale + y_mean
    test_metadata = metadata.iloc[test_rows].reset_index(drop=True)
    aggregated = aggregate_rows_by_participant(
        row_prediction, test_metadata["participant_id"].astype(str).to_numpy()
    )
    return aggregated, best["settings"], records


def aggregate_rows_by_participant(values, participant_ids):
    output = {}
    for participant_id in sorted(set(participant_ids)):
        selection = participant_ids == participant_id
        output[str(participant_id)] = values[selection].mean(axis=0)
    return output


def build_stage1_difference(stage1, config, data, metadata, device, batch_size):
    model_cfg = config["model"]
    clinical = ClinicalEncoder(
        len(stage1["feature_names"]),
        model_cfg["clinical_hidden"],
        model_cfg["clinical_layers"],
        model_cfg["clinical_heads"],
        model_cfg["clinical_dropout"],
    )
    head = ProteinPredictionHead(
        model_cfg["clinical_hidden"],
        len(stage1["protein_names"]),
        model_cfg["clinical_dropout"],
    )
    encoder_state = {
        key[len("encoder.") :]: value
        for key, value in stage1["model_state"].items()
        if key.startswith("encoder.")
    }
    head_state = {
        key[len("head.") :]: value
        for key, value in stage1["model_state"].items()
        if key.startswith("head.")
    }
    clinical.load_state_dict(encoder_state)
    head.load_state_dict(head_state)
    clinical = clinical.to(device).eval()
    head = head.to(device).eval()
    protein_scale = np.asarray(
        stage1["protein_scaler"]["scale"], dtype=np.float32
    )
    row_predictions = []
    with torch.no_grad():
        for start in range(0, len(metadata), batch_size):
            pre = torch.from_numpy(
                data["x_pre"][start : start + batch_size]
            ).to(device)
            post = torch.from_numpy(
                data["x_post"][start : start + batch_size]
            ).to(device)
            pre_prediction = head(clinical(pre))
            post_prediction = head(clinical(post))
            delta = (post_prediction - pre_prediction).cpu().numpy()
            row_predictions.append(delta * protein_scale)
    row_predictions = np.concatenate(row_predictions, axis=0)
    return aggregate_rows_by_participant(
        row_predictions,
        metadata["participant_id"].astype(str).to_numpy(),
    )


def read_wide_prediction(path, id_column, protein_names):
    frame = pd.read_csv(path, dtype={id_column: str})
    missing = [name for name in protein_names if name not in frame.columns]
    if missing:
        raise ValueError(
            "Prediction file is missing %d proteins, including %s"
            % (len(missing), missing[:5])
        )
    frame[id_column] = frame[id_column].astype(str)
    return frame.set_index(id_column)[protein_names]


def pearson_correlation(observed, predicted):
    if len(observed) < 3:
        return np.nan
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = observed - observed.mean()
    predicted = predicted - predicted.mean()
    denominator = math.sqrt(
        float(np.dot(observed, observed) * np.dot(predicted, predicted))
    )
    if denominator <= 1e-12:
        return np.nan
    return float(np.dot(observed, predicted) / denominator)


def spearman_correlation(observed, predicted):
    if len(observed) < 3:
        return np.nan
    observed_rank = pd.Series(observed).rank(method="average").to_numpy()
    predicted_rank = pd.Series(predicted).rank(method="average").to_numpy()
    return pearson_correlation(observed_rank, predicted_rank)


def vector_metrics(observed, predicted):
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid].astype(np.float64)
    predicted = predicted[valid].astype(np.float64)
    n = len(observed)
    if n == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "r2_score": np.nan,
            "pearson": np.nan,
            "spearman": np.nan,
        }
    error = predicted - observed
    denominator = np.sum((observed - observed.mean()) ** 2)
    r2 = (
        1.0 - np.sum(error**2) / denominator
        if denominator > 1e-12
        else np.nan
    )
    return {
        "n": int(n),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2_score": float(r2),
        "pearson": pearson_correlation(observed, predicted),
        "spearman": spearman_correlation(observed, predicted),
    }


def per_protein_metrics(model_name, observed, predicted, protein_names):
    rows = []
    for index, protein in enumerate(protein_names):
        metrics = vector_metrics(observed[:, index], predicted[:, index])
        metrics.update({"model": model_name, "protein": protein})
        rows.append(metrics)
    return pd.DataFrame(rows)


def per_subject_metrics(model_name, observed, predicted, participant_ids):
    rows = []
    for index, participant_id in enumerate(participant_ids):
        metrics = vector_metrics(observed[index], predicted[index])
        metrics.update(
            {"model": model_name, "participant_id": str(participant_id)}
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def summarize_model(
    model_name,
    observed,
    predicted,
    protein_frame,
    subject_frame,
    min_protein_n,
):
    pooled = vector_metrics(observed.reshape(-1), predicted.reshape(-1))
    eligible = protein_frame[protein_frame["n"] >= min_protein_n]
    result = {
        "model": model_name,
        "model_label": MODEL_LABELS[model_name],
        "n_observed_cells": pooled["n"],
        "pooled_mae": pooled["mae"],
        "pooled_rmse": pooled["rmse"],
        "pooled_r2_score": pooled["r2_score"],
        "pooled_pearson": pooled["pearson"],
        "n_proteins_total": int(len(protein_frame)),
        "n_proteins_eligible": int(len(eligible)),
        "minimum_n_for_protein_summary": int(min_protein_n),
        "median_protein_pearson": float(eligible["pearson"].median()),
        "median_protein_spearman": float(eligible["spearman"].median()),
        "median_protein_r2_score": float(eligible["r2_score"].median()),
        "proteins_pearson_gt_0": int((eligible["pearson"] > 0).sum()),
        "proteins_pearson_ge_0_1": int((eligible["pearson"] >= 0.1).sum()),
        "proteins_pearson_ge_0_2": int((eligible["pearson"] >= 0.2).sum()),
        "proteins_pearson_ge_0_3": int((eligible["pearson"] >= 0.3).sum()),
        "proteins_spearman_gt_0": int((eligible["spearman"] > 0).sum()),
        "proteins_r2_score_gt_0": int((eligible["r2_score"] > 0).sum()),
        "median_subject_mae": float(subject_frame["mae"].median()),
        "median_subject_rmse": float(subject_frame["rmse"].median()),
        "median_subject_r2_score": float(subject_frame["r2_score"].median()),
        "median_subject_pearson": float(subject_frame["pearson"].median()),
        "median_subject_spearman": float(subject_frame["spearman"].median()),
    }
    return result


def fold_metrics(
    model_name,
    fold,
    fold_ids,
    all_ids,
    observed,
    predicted,
    protein_names,
    min_protein_n,
):
    index_lookup = {value: index for index, value in enumerate(all_ids)}
    indices = np.asarray([index_lookup[value] for value in fold_ids], dtype=int)
    fold_observed = observed[indices]
    fold_predicted = predicted[indices]
    protein = per_protein_metrics(
        model_name, fold_observed, fold_predicted, protein_names
    )
    subject = per_subject_metrics(
        model_name, fold_observed, fold_predicted, fold_ids
    )
    summary = summarize_model(
        model_name,
        fold_observed,
        fold_predicted,
        protein,
        subject,
        min_protein_n,
    )
    summary.update({"fold": int(fold), "n_participants": int(len(indices))})
    return summary


def write_prediction(path, participant_ids, protein_names, values, id_column):
    frame = pd.DataFrame(values, columns=protein_names)
    frame.insert(0, id_column, participant_ids)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    if args.fast:
        args.max_epochs_linear = min(args.max_epochs_linear, 60)
        args.max_epochs_mlp = min(args.max_epochs_mlp, 80)
        args.patience = min(args.patience, 12)

    output_dir = (
        Path(args.output).resolve()
        if args.output
        else Path(config["output_dir"]) / "stage3" / "benchmark_results"
    )
    predictions_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    stage1_path = Path(config["output_dir"]) / "stage1" / "best.pt"
    full_prediction_path = (
        Path(args.full_predictions).resolve()
        if args.full_predictions
        else Path(config["output_dir"])
        / "stage3"
        / "oof_results"
        / "predicted_delta_protein.csv"
    )
    if not stage1_path.exists():
        raise FileNotFoundError("Stage 1 checkpoint not found: %s" % stage1_path)
    if not full_prediction_path.exists():
        raise FileNotFoundError(
            "Full-model OOF predictions not found: %s" % full_prediction_path
        )

    print("Device:", device)
    print("Loading Stage 1 checkpoint and longitudinal data...")
    stage1 = torch_load(stage1_path, map_location="cpu")
    data = _load_longitudinal(config, stage1)
    metadata = data["metadata"].copy()
    metadata["participant_id"] = metadata["participant_id"].astype(str)
    participant_ids = np.asarray(sorted(metadata["participant_id"].unique()))
    groups = metadata["participant_id"].to_numpy()
    protein_names = list(stage1["protein_names"])
    id_column = config["data"]["id_column"]
    first_rows = (
        metadata.reset_index()
        .drop_duplicates("participant_id")
        .set_index("participant_id")
        .loc[participant_ids]["index"]
        .to_numpy()
    )
    true_delta = data["true_delta"][first_rows].astype(np.float32)
    features = feature_matrix(data)

    splitter = KFold(
        n_splits=config["stage3"]["folds"],
        shuffle=True,
        random_state=config["seed"],
    )
    fold_definitions = []
    fold_assignment_rows = []
    for fold, (outer_train, outer_test) in enumerate(
        splitter.split(participant_ids), start=1
    ):
        train_ids = participant_ids[outer_train]
        test_ids = participant_ids[outer_test]
        fold_definitions.append((fold, train_ids, test_ids))
        fold_assignment_rows.extend(
            {"participant_id": value, "fold": fold} for value in test_ids
        )
    pd.DataFrame(fold_assignment_rows).to_csv(
        output_dir / "outer_fold_assignment.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_predictions = {
        model: np.full_like(true_delta, np.nan, dtype=np.float32)
        for model in MODEL_ORDER
    }
    full_frame = read_wide_prediction(
        full_prediction_path, id_column, protein_names
    )
    missing_full_ids = [
        value for value in participant_ids if value not in full_frame.index
    ]
    if missing_full_ids:
        raise ValueError(
            "Full prediction file is missing participant IDs: %s"
            % missing_full_ids[:10]
        )
    all_predictions["full_model"] = full_frame.loc[
        participant_ids, protein_names
    ].to_numpy(dtype=np.float32)

    print("Computing Stage 1 cross-sectional difference baseline...")
    stage1_difference = build_stage1_difference(
        stage1, config, data, metadata, device, args.batch_size
    )
    all_predictions["stage1_cross_sectional_difference"] = np.stack(
        [stage1_difference[value] for value in participant_ids]
    ).astype(np.float32)

    ridge_settings = [
        {"alpha": value, "l1_ratio": 0.0}
        for value in parse_float_list(args.ridge_alphas)
    ]
    elastic_settings = parse_elastic_settings(args.elasticnet_settings)
    if args.fast:
        ridge_settings = ridge_settings[:1]
        elastic_settings = elastic_settings[:1]
    mlp_settings = [{"hidden": args.mlp_hidden, "dropout": args.mlp_dropout}]
    candidate_records = []
    id_to_output_index = {
        participant_id: index
        for index, participant_id in enumerate(participant_ids)
    }

    for fold, train_ids, test_ids in fold_definitions:
        start_time = time.time()
        print(
            "Fold %d/%d: %d train, %d test participants"
            % (fold, len(fold_definitions), len(train_ids), len(test_ids))
        )
        outer_train_rows = np.where(np.isin(groups, train_ids))[0]
        test_rows = np.where(np.isin(groups, test_ids))[0]
        train_rows, val_rows = inner_split(
            outer_train_rows, groups, config
        )
        train_mask = np.isfinite(data["true_delta"][outer_train_rows])
        train_mean, _ = masked_mean_scale(
            data["true_delta"][outer_train_rows], train_mask
        )
        for participant_id in test_ids:
            output_index = id_to_output_index[participant_id]
            all_predictions["zero_change"][output_index] = 0.0
            all_predictions["training_mean_delta"][output_index] = train_mean

        baseline_specs = [
            ("ridge", ridge_settings),
            ("elastic_net", elastic_settings),
            ("plain_mlp", mlp_settings),
        ]
        for model_offset, (kind, settings) in enumerate(baseline_specs):
            print("  Training %s (%d candidate setting(s))..." % (kind, len(settings)))
            aggregated, selected, candidates = fit_supervised_baseline(
                kind,
                settings,
                features,
                data["true_delta"],
                train_rows,
                val_rows,
                test_rows,
                metadata,
                args,
                device,
                config["seed"] + fold * 100 + model_offset * 10,
            )
            for participant_id in test_ids:
                all_predictions[kind][id_to_output_index[participant_id]] = (
                    aggregated[participant_id]
                )
            for candidate_index, candidate in enumerate(candidates):
                row = {
                    "fold": fold,
                    "model": kind,
                    "candidate_index": candidate_index,
                    "selected": candidate_index
                    == int(np.argmin(
                        [item["validation_standardized_mse"] for item in candidates]
                    )),
                }
                row.update(candidate)
                candidate_records.append(row)
            print("    Selected:", selected)
        print("  Fold elapsed: %.1f min" % ((time.time() - start_time) / 60.0))

    for model_name, prediction in all_predictions.items():
        if not np.isfinite(prediction).all():
            count = int((~np.isfinite(prediction)).sum())
            raise RuntimeError(
                "%s contains %d non-finite predictions." % (model_name, count)
            )
        write_prediction(
            predictions_dir / ("predicted_delta_%s.csv" % safe_name(model_name)),
            participant_ids,
            protein_names,
            prediction,
            id_column,
        )

    print("Calculating pooled, per-protein, per-subject, and per-fold metrics...")
    protein_frames = []
    subject_frames = []
    summary_rows = []
    fold_rows = []
    for model_name in MODEL_ORDER:
        prediction = all_predictions[model_name]
        protein = per_protein_metrics(
            model_name, true_delta, prediction, protein_names
        )
        subject = per_subject_metrics(
            model_name, true_delta, prediction, participant_ids
        )
        protein_frames.append(protein)
        subject_frames.append(subject)
        summary_rows.append(
            summarize_model(
                model_name,
                true_delta,
                prediction,
                protein,
                subject,
                args.min_protein_n,
            )
        )
        for fold, _, test_ids in fold_definitions:
            fold_rows.append(
                fold_metrics(
                    model_name,
                    fold,
                    test_ids,
                    participant_ids.tolist(),
                    true_delta,
                    prediction,
                    protein_names,
                    args.min_protein_n,
                )
            )

    summary_frame = pd.DataFrame(summary_rows)
    protein_frame = pd.concat(protein_frames, ignore_index=True)
    subject_frame = pd.concat(subject_frames, ignore_index=True)
    fold_frame = pd.DataFrame(fold_rows)
    summary_frame.to_csv(
        output_dir / "baseline_comparison_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    protein_frame.to_csv(
        output_dir / "metrics_per_protein_all_models.csv",
        index=False,
        encoding="utf-8-sig",
    )
    subject_frame.to_csv(
        output_dir / "metrics_per_subject_all_models.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fold_frame.to_csv(
        output_dir / "metrics_per_fold_all_models.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(candidate_records).to_csv(
        output_dir / "baseline_hyperparameter_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_prediction(
        output_dir / "true_delta_protein_aligned.csv",
        participant_ids,
        protein_names,
        true_delta,
        id_column,
    )

    summary_payload = {
        "evaluation": (
            "Participant-level 5-fold out-of-fold Stage 3 comparison. All models "
            "use the identical outer test folds."
        ),
        "n_participants": int(len(participant_ids)),
        "n_proteins": int(len(protein_names)),
        "n_observed_delta_cells": int(np.isfinite(true_delta).sum()),
        "missing_delta_fraction": float(np.mean(~np.isfinite(true_delta))),
        "minimum_n_for_primary_per_protein_summary": int(args.min_protein_n),
        "models": summary_rows,
        "metric_definitions": {
            "pooled_r2_score": (
                "1-SSE/SST after pooling all observed participant-protein cells."
            ),
            "protein_metrics": (
                "Calculated separately across participants for each protein. "
                "Primary medians/counts include proteins with n >= the configured "
                "minimum."
            ),
            "subject_metrics": (
                "Calculated separately across observed proteins within each participant."
            ),
            "stage1_cross_sectional_difference": (
                "Stage 1 protein prediction at post minus Stage 1 protein prediction "
                "at pre, averaged across imputation versions."
            ),
        },
        "leakage_control": (
            "Splits are reconstructed with the same sorted participant IDs, KFold "
            "settings, fold count, shuffle flag, and random seed used by Stage 3. "
            "No participant appears in both training and test data within a fold."
        ),
    }
    save_json(summary_payload, output_dir / "benchmark_metrics_summary.json")
    print("Finished. Results:", output_dir)


if __name__ == "__main__":
    main()
