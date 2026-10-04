from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .data import (
    CrossSectionalDataset,
    Standardizer,
    longitudinal_paths,
    read_clinical,
    read_olink,
    stage1_paths,
)
from .losses import masked_huber, masked_mse
from .models import ClinicalEncoder, CrossSectionalModel, ProteinPredictionHead
from .utils import EarlyStopping, atomic_torch_save, save_json


def build_stage1_model(config: Dict, n_features: int, n_proteins: int):
    model_cfg = config["model"]
    encoder = ClinicalEncoder(
        n_features=n_features,
        hidden=model_cfg["clinical_hidden"],
        layers=model_cfg["clinical_layers"],
        heads=model_cfg["clinical_heads"],
        dropout=model_cfg["clinical_dropout"],
    )
    head = ProteinPredictionHead(
        hidden=model_cfg["clinical_hidden"],
        n_proteins=n_proteins,
        dropout=model_cfg["clinical_dropout"],
    )
    return CrossSectionalModel(encoder, head)


def _load_stage1_frame(config: Dict):
    root = Path(config["data"]["root"])
    clinical_path, olink_path = stage1_paths(
        root, config["data"]["stage1_imputation"]
    )
    max_rows = config["data"].get("max_stage1_rows")
    id_column = config["data"]["id_column"]
    clinical = read_clinical(clinical_path, id_column)
    if config["data"].get("exclude_longitudinal_participants_from_stage1", True):
        longitudinal_pre = read_clinical(
            longitudinal_paths(root, "LabTOP")[0], id_column
        )
        excluded_ids = set(longitudinal_pre[id_column].astype(str))
        clinical = clinical[~clinical[id_column].isin(excluded_ids)].copy()
    if max_rows is not None:
        clinical = clinical.iloc[: int(max_rows)].copy()
    olink = read_olink(
        olink_path,
        id_column,
        nrows=int(max_rows) if max_rows is not None else None,
    )
    frame = clinical.merge(olink, on=id_column, how="inner", validate="one_to_one")
    feature_names = [column for column in clinical.columns if column != id_column]
    protein_names = [column for column in olink.columns if column != id_column]
    return frame, feature_names, protein_names


def _predict_stage1(
    model: CrossSectionalModel,
    x_scaled: np.ndarray,
    protein_scaler: Standardizer,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x_scaled), batch_size):
            batch = torch.from_numpy(
                x_scaled[start : start + batch_size].astype(np.float32)
            ).to(device)
            prediction = model(batch).cpu().numpy()
            predictions.append(prediction)
    standardized = np.concatenate(predictions, axis=0)
    return protein_scaler.inverse_transform(standardized).astype(np.float32)


def _per_protein_cross_sectional_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
    protein_names,
) -> pd.DataFrame:
    rows = []
    for index, protein in enumerate(protein_names):
        valid = mask[:, index]
        y = observed[valid, index]
        p = predicted[valid, index]
        pearson = float("nan")
        spearman = float("nan")
        predictive_r2 = float("nan")
        conventional_r2 = float("nan")
        if len(y) >= 3 and np.ptp(y) > 1e-8 and np.ptp(p) > 1e-8:
            pearson = float(pearsonr(y, p)[0])
            spearman = float(spearmanr(y, p).correlation)
            predictive_r2 = pearson**2
            conventional_r2 = float(r2_score(y, p))
        rows.append(
            {
                "protein": protein,
                "n_observed": int(valid.sum()),
                "pearson_r": pearson,
                "predictive_r2_squared_pearson": predictive_r2,
                "spearman_rho": spearman,
                "r2_score": conventional_r2,
                "mae": float(np.mean(np.abs(p - y))) if len(y) else float("nan"),
                "rmse": float(np.sqrt(np.mean((p - y) ** 2)))
                if len(y)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def export_stage1_predictions(
    config: Dict,
    device: torch.device,
    logger,
    checkpoint_path: Optional[Path] = None,
) -> Path:
    output = Path(config["output_dir"]) / "stage1" / "cross_sectional_results"
    output.mkdir(parents=True, exist_ok=True)
    if checkpoint_path is None:
        checkpoint_path = Path(config["output_dir"]) / "stage1" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    frame, feature_names, protein_names = _load_stage1_frame(config)
    if feature_names != checkpoint["feature_names"]:
        raise ValueError("Clinical feature schema differs from Stage 1 checkpoint.")
    if protein_names != checkpoint["protein_names"]:
        raise ValueError("Protein schema differs from Stage 1 checkpoint.")

    id_column = config["data"]["id_column"]
    indices = np.arange(len(frame))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=config["stage1"]["val_fraction"],
        random_state=config["seed"],
    )
    clinical_scaler = Standardizer.from_state_dict(
        checkpoint["clinical_scaler"]
    )
    protein_scaler = Standardizer.from_state_dict(checkpoint["protein_scaler"])
    x = frame[feature_names].to_numpy(dtype=np.float32)
    observed = frame[protein_names].to_numpy(dtype=np.float32)
    observed_mask = np.isfinite(observed)
    x_scaled = clinical_scaler.transform(x)

    model = build_stage1_model(config, len(feature_names), len(protein_names)).to(
        device
    )
    model.load_state_dict(checkpoint["model_state"])
    predicted = _predict_stage1(
        model,
        x_scaled,
        protein_scaler,
        config["stage1"]["batch_size"],
        device,
    )

    split = np.full(len(frame), "train", dtype=object)
    split[val_idx] = "validation"
    split_frame = pd.DataFrame(
        {
            id_column: frame[id_column].astype(str).to_numpy(),
            "split": split,
        }
    )
    split_frame.to_csv(
        output / "cross_sectional_split.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_prediction_frame = pd.DataFrame(predicted, columns=protein_names)
    all_prediction_frame.insert(
        0, id_column, frame[id_column].astype(str).to_numpy()
    )
    all_prediction_frame.insert(1, "split", split)
    all_prediction_frame.to_csv(
        output / "predicted_cross_sectional_proteome_all.csv",
        index=False,
        encoding="utf-8-sig",
    )

    validation_prediction = pd.DataFrame(
        predicted[val_idx], columns=protein_names
    )
    validation_prediction.insert(
        0, id_column, frame.iloc[val_idx][id_column].astype(str).to_numpy()
    )
    validation_prediction.to_csv(
        output / "predicted_cross_sectional_proteome_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    validation_observed = pd.DataFrame(
        observed[val_idx], columns=protein_names
    )
    validation_observed.insert(
        0, id_column, frame.iloc[val_idx][id_column].astype(str).to_numpy()
    )
    validation_observed.to_csv(
        output / "observed_cross_sectional_proteome_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics = _per_protein_cross_sectional_metrics(
        observed[val_idx],
        predicted[val_idx],
        observed_mask[val_idx],
        protein_names,
    )
    metrics.to_csv(
        output / "cross_sectional_validation_metrics_per_protein.csv",
        index=False,
        encoding="utf-8-sig",
    )
    valid_r2 = metrics["predictive_r2_squared_pearson"].dropna()
    micro_mask = observed_mask[val_idx]
    micro_error = predicted[val_idx][micro_mask] - observed[val_idx][micro_mask]
    summary = {
        "n_total_participants": int(len(frame)),
        "n_training_participants": int(len(train_idx)),
        "n_validation_participants": int(len(val_idx)),
        "n_proteins": int(len(protein_names)),
        "micro_mae": float(np.mean(np.abs(micro_error))),
        "micro_rmse": float(np.sqrt(np.mean(micro_error**2))),
        "median_predictive_r2_squared_pearson": float(valid_r2.median()),
        "mean_predictive_r2_squared_pearson": float(valid_r2.mean()),
        "proteins_predictive_r2_gt_0_01": int((valid_r2 > 0.01).sum()),
        "proteins_predictive_r2_gt_0_05": int((valid_r2 > 0.05).sum()),
        "proteins_predictive_r2_gt_0_10": int((valid_r2 > 0.10).sum()),
        "evaluation_note": (
            "Only the held-out validation predictions are valid for performance "
            "comparison. The all-participant file includes training-set fitted "
            "predictions and must not be used to claim out-of-sample accuracy."
        ),
        "literature_metric_note": (
            "predictive_r2_squared_pearson is squared Pearson correlation, matching "
            "the predictive R2 definition used by common genetic protein-imputation "
            "studies. r2_score is also reported separately and is not interchangeable."
        ),
    }
    save_json(
        summary,
        output / "cross_sectional_validation_metrics_summary.json",
    )
    logger.info(
        "Stage 1 cross-sectional exports: %d all-participant predictions, "
        "%d held-out validation predictions -> %s",
        len(frame),
        len(val_idx),
        output,
    )
    return output


def export_stage1_oof_predictions(
    config: Dict,
    device: torch.device,
    logger,
    folds: int = 5,
) -> Path:
    if folds < 2:
        raise ValueError("OOF evaluation requires at least two folds.")
    output = (
        Path(config["output_dir"])
        / "stage1"
        / "cross_sectional_oof_results"
    )
    output.mkdir(parents=True, exist_ok=True)
    frame, feature_names, protein_names = _load_stage1_frame(config)
    id_column = config["data"]["id_column"]
    x = frame[feature_names].to_numpy(dtype=np.float32)
    observed = frame[protein_names].to_numpy(dtype=np.float32)
    observed_mask = np.isfinite(observed)
    oof_prediction = np.full_like(observed, np.nan, dtype=np.float32)
    fold_assignment = np.full(len(frame), -1, dtype=np.int32)
    splitter = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=config["seed"],
    )
    loss_function = (
        masked_huber
        if config["stage1"].get("loss", "huber") == "huber"
        else masked_mse
    )

    for fold, (outer_train_idx, test_idx) in enumerate(
        splitter.split(x), start=1
    ):
        train_idx, val_idx = train_test_split(
            outer_train_idx,
            test_size=config["stage1"]["val_fraction"],
            random_state=config["seed"] + fold,
        )
        x_scaler = Standardizer.fit(x[train_idx])
        y_scaler = Standardizer.fit(observed[train_idx])
        x_train = x_scaler.transform(x[train_idx])
        x_val = x_scaler.transform(x[val_idx])
        x_test = x_scaler.transform(x[test_idx])
        y_standardized = (
            (observed - y_scaler.mean) / y_scaler.scale
        ).astype(np.float32)
        train_data = CrossSectionalDataset(
            x_train,
            y_standardized[train_idx],
            observed_mask[train_idx],
        )
        val_data = CrossSectionalDataset(
            x_val,
            y_standardized[val_idx],
            observed_mask[val_idx],
        )
        train_loader = DataLoader(
            train_data,
            batch_size=config["stage1"]["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            val_data,
            batch_size=config["stage1"]["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=device.type == "cuda",
        )
        model = build_stage1_model(
            config, len(feature_names), len(protein_names)
        ).to(device)
        optimizer = AdamW(
            model.parameters(),
            lr=config["stage1"]["learning_rate"],
            weight_decay=config["stage1"]["weight_decay"],
        )
        scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        amp_scaler = GradScaler(enabled=device.type == "cuda")
        stopping = EarlyStopping(config["stage1"]["patience"])
        fold_checkpoint = output / ("fold_%02d_best.pt" % fold)
        logger.info(
            "Stage 1 OOF fold %d/%d | train %d | validation %d | test %d",
            fold,
            folds,
            len(train_idx),
            len(val_idx),
            len(test_idx),
        )
        for epoch in range(1, config["stage1"]["epochs"] + 1):
            model.train()
            train_sum = 0.0
            train_count = 0
            for batch_x, batch_y, batch_mask in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_mask = batch_mask.to(device)
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=device.type == "cuda"):
                    prediction = model(batch_x)
                    loss = loss_function(prediction, batch_y, batch_mask)
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()
                train_sum += loss.item() * batch_x.shape[0]
                train_count += batch_x.shape[0]
            model.eval()
            val_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for batch_x, batch_y, batch_mask in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    batch_mask = batch_mask.to(device)
                    prediction = model(batch_x)
                    loss = loss_function(prediction, batch_y, batch_mask)
                    val_sum += loss.item() * batch_x.shape[0]
                    val_count += batch_x.shape[0]
            train_loss = train_sum / max(train_count, 1)
            val_loss = val_sum / max(val_count, 1)
            scheduler.step(val_loss)
            improved = stopping.update(val_loss)
            logger.info(
                "Stage 1 OOF fold %d epoch %d | train %.6f | val %.6f",
                fold,
                epoch,
                train_loss,
                val_loss,
            )
            if improved:
                atomic_torch_save(
                    {
                        "stage": "1_oof",
                        "fold": fold,
                        "model_state": model.state_dict(),
                        "feature_names": feature_names,
                        "protein_names": protein_names,
                        "clinical_scaler": x_scaler.state_dict(),
                        "protein_scaler": y_scaler.state_dict(),
                        "best_val_loss": val_loss,
                    },
                    fold_checkpoint,
                )
            if stopping.should_stop:
                break
        best = torch.load(fold_checkpoint, map_location=device)
        model.load_state_dict(best["model_state"])
        oof_prediction[test_idx] = _predict_stage1(
            model,
            x_test,
            y_scaler,
            config["stage1"]["batch_size"],
            device,
        )
        fold_assignment[test_idx] = fold
        del model, optimizer, train_loader, val_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if np.isnan(oof_prediction).any() or np.any(fold_assignment < 1):
        raise RuntimeError("OOF prediction matrix is incomplete.")
    prediction_frame = pd.DataFrame(oof_prediction, columns=protein_names)
    prediction_frame.insert(
        0, id_column, frame[id_column].astype(str).to_numpy()
    )
    prediction_frame.insert(1, "oof_fold", fold_assignment)
    prediction_frame.to_csv(
        output / "predicted_cross_sectional_proteome_oof_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        {
            id_column: frame[id_column].astype(str).to_numpy(),
            "oof_fold": fold_assignment,
        }
    ).to_csv(
        output / "cross_sectional_oof_fold_assignment.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics = _per_protein_cross_sectional_metrics(
        observed,
        oof_prediction,
        observed_mask,
        protein_names,
    )
    metrics.to_csv(
        output / "cross_sectional_oof_metrics_per_protein.csv",
        index=False,
        encoding="utf-8-sig",
    )
    valid_r2 = metrics["predictive_r2_squared_pearson"].dropna()
    micro_error = oof_prediction[observed_mask] - observed[observed_mask]
    save_json(
        {
            "evaluation": "%d-fold out-of-fold cross-sectional prediction" % folds,
            "n_participants": int(len(frame)),
            "n_proteins": int(len(protein_names)),
            "micro_mae": float(np.mean(np.abs(micro_error))),
            "micro_rmse": float(np.sqrt(np.mean(micro_error**2))),
            "median_predictive_r2_squared_pearson": float(valid_r2.median()),
            "mean_predictive_r2_squared_pearson": float(valid_r2.mean()),
            "proteins_predictive_r2_gt_0_01": int((valid_r2 > 0.01).sum()),
            "proteins_predictive_r2_gt_0_05": int((valid_r2 > 0.05).sum()),
            "proteins_predictive_r2_gt_0_10": int((valid_r2 > 0.10).sum()),
            "primary_comparison_metric": "predictive_r2_squared_pearson",
            "comparison_note": (
                "Every participant prediction was generated by a model that did "
                "not train on that participant. Use this OOF result for internal "
                "baseline comparison and manuscript performance reporting."
            ),
        },
        output / "cross_sectional_oof_metrics_summary.json",
    )
    logger.info(
        "Stage 1 %d-fold OOF export completed: %d participants -> %s",
        folds,
        len(frame),
        output,
    )
    return output


def train_stage1(config: Dict, device: torch.device, logger) -> Path:
    output = Path(config["output_dir"]) / "stage1"
    checkpoint_path = output / "best.pt"
    frame, feature_names, protein_names = _load_stage1_frame(config)
    id_column = config["data"]["id_column"]
    x = frame[feature_names].to_numpy(dtype=np.float32)
    y = frame[protein_names].to_numpy(dtype=np.float32)
    mask = np.isfinite(y)
    indices = np.arange(len(frame))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=config["stage1"]["val_fraction"],
        random_state=config["seed"],
    )
    x_scaler = Standardizer.fit(x[train_idx])
    y_scaler = Standardizer.fit(y[train_idx])
    x_scaled = x_scaler.transform(x)
    y_scaled = ((y - y_scaler.mean) / y_scaler.scale).astype(np.float32)
    train_data = CrossSectionalDataset(
        x_scaled[train_idx], y_scaled[train_idx], mask[train_idx]
    )
    val_data = CrossSectionalDataset(
        x_scaled[val_idx], y_scaled[val_idx], mask[val_idx]
    )
    train_loader = DataLoader(
        train_data,
        batch_size=config["stage1"]["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config["stage1"]["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
    )
    model = build_stage1_model(config, len(feature_names), len(protein_names)).to(
        device
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config["stage1"]["learning_rate"],
        weight_decay=config["stage1"]["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    amp_scaler = GradScaler(enabled=device.type == "cuda")
    stopping = EarlyStopping(config["stage1"]["patience"])
    loss_function = (
        masked_huber if config["stage1"].get("loss", "huber") == "huber" else masked_mse
    )
    logger.info(
        "Stage 1: %d samples, %d clinical features, %d proteins",
        len(frame),
        len(feature_names),
        len(protein_names),
    )
    for epoch in range(1, config["stage1"]["epochs"] + 1):
        model.train()
        train_sum = 0.0
        train_count = 0
        for batch_x, batch_y, batch_mask in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_mask = batch_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                prediction = model(batch_x)
                loss = loss_function(prediction, batch_y, batch_mask)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            train_sum += loss.item() * batch_x.shape[0]
            train_count += batch_x.shape[0]
        model.eval()
        val_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_x, batch_y, batch_mask in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_mask = batch_mask.to(device)
                prediction = model(batch_x)
                loss = loss_function(prediction, batch_y, batch_mask)
                val_sum += loss.item() * batch_x.shape[0]
                val_count += batch_x.shape[0]
        train_loss = train_sum / max(train_count, 1)
        val_loss = val_sum / max(val_count, 1)
        scheduler.step(val_loss)
        improved = stopping.update(val_loss)
        logger.info(
            "Stage 1 epoch %d | train %.6f | val %.6f",
            epoch,
            train_loss,
            val_loss,
        )
        if improved:
            atomic_torch_save(
                {
                    "stage": 1,
                    "model_state": model.state_dict(),
                    "feature_names": feature_names,
                    "protein_names": protein_names,
                    "clinical_scaler": x_scaler.state_dict(),
                    "protein_scaler": y_scaler.state_dict(),
                    "model_config": config["model"],
                    "best_val_loss": val_loss,
                },
                checkpoint_path,
            )
        if stopping.should_stop:
            break
    export_stage1_predictions(
        config, device, logger, checkpoint_path=checkpoint_path
    )
    return checkpoint_path
