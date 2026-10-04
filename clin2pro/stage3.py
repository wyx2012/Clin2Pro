from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, KFold
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .data import (
    Standardizer,
    TemporalDataset,
    load_drug_profiles,
    load_pathway_matrices,
    longitudinal_paths,
    read_clinical,
    read_olink,
)
from .evaluation import export_drug_similarity, export_evaluation
from .losses import masked_huber, masked_mse, masked_pathway_profiles
from .models import (
    ClinicalEncoder,
    LongitudinalModel,
    ProteinPredictionHead,
    TemporalPerturbationEncoder,
)
from .utils import EarlyStopping, atomic_torch_save, parameter_groups


def _state_with_prefix(state: Dict, prefix: str) -> Dict:
    return {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }


def _load_longitudinal(config: Dict, stage1: Dict):
    root = Path(config["data"]["root"])
    id_column = config["data"]["id_column"]
    feature_names = stage1["feature_names"]
    protein_names = stage1["protein_names"]
    scaler = Standardizer.from_state_dict(stage1["clinical_scaler"])
    first_paths = longitudinal_paths(
        root, config["data"]["stage3_imputations"][0]
    )
    protein_pre = read_olink(first_paths[2], id_column)
    protein_post = read_olink(first_paths[3], id_column)
    protein = protein_pre.merge(
        protein_post,
        on=id_column,
        suffixes=("_pre", "_post"),
        validate="one_to_one",
    )
    protein["_true_delta"] = list(
        protein[[name + "_post" for name in protein_names]].to_numpy(
            dtype=np.float32
        )
        - protein[[name + "_pre" for name in protein_names]].to_numpy(
            dtype=np.float32
        )
    )
    protein_lookup = protein.set_index(id_column)["_true_delta"].to_dict()
    rows = []
    x_pre_all = []
    x_post_all = []
    interval_all = []
    true_all = []
    for imputation in config["data"]["stage3_imputations"]:
        pre_path, post_path, _, _ = longitudinal_paths(root, imputation)
        pre = read_clinical(pre_path, id_column)
        post = read_clinical(post_path, id_column)
        paired = pre.merge(
            post,
            on=id_column,
            suffixes=("_pre", "_post"),
            validate="one_to_one",
        )
        paired = paired[paired[id_column].isin(protein_lookup)].copy()
        interval = (
            paired["age_at_instance_post"] - paired["age_at_instance_pre"]
        ).to_numpy(dtype=np.float32)
        valid = np.isfinite(interval) & (interval > 0)
        paired = paired.loc[valid].reset_index(drop=True)
        interval = interval[valid]
        x_pre = paired[[name + "_pre" for name in feature_names]].to_numpy(
            dtype=np.float32
        )
        x_post = paired[[name + "_post" for name in feature_names]].to_numpy(
            dtype=np.float32
        )
        true = np.stack(
            [protein_lookup[str(value)] for value in paired[id_column]], axis=0
        ).astype(np.float32)
        x_pre_all.append(scaler.transform(x_pre))
        x_post_all.append(scaler.transform(x_post))
        interval_all.append(interval)
        true_all.append(true)
        rows.extend(
            {
                "participant_id": str(participant_id),
                "imputation": imputation,
            }
            for participant_id in paired[id_column]
        )
    metadata = pd.DataFrame(rows)
    return {
        "metadata": metadata,
        "x_pre": np.concatenate(x_pre_all),
        "x_post": np.concatenate(x_post_all),
        "interval": np.concatenate(interval_all),
        "true_delta": np.concatenate(true_all),
    }


def _build_model(config: Dict, stage1: Dict, stage2: Dict, device):
    model_cfg = config["model"]
    clinical = ClinicalEncoder(
        len(stage1["feature_names"]),
        model_cfg["clinical_hidden"],
        model_cfg["clinical_layers"],
        model_cfg["clinical_heads"],
        model_cfg["clinical_dropout"],
    )
    protein_head = ProteinPredictionHead(
        model_cfg["clinical_hidden"],
        len(stage1["protein_names"]),
        model_cfg["clinical_dropout"],
    )
    temporal = TemporalPerturbationEncoder(
        model_cfg["clinical_hidden"],
        len(stage1["feature_names"]),
        model_cfg["temporal_hidden"],
        model_cfg["temporal_dropout"],
    )
    clinical.load_state_dict(
        _state_with_prefix(stage2["model_state"], "clinical_encoder.")
    )
    temporal.load_state_dict(
        _state_with_prefix(stage2["model_state"], "temporal_encoder.")
    )
    protein_head.load_state_dict(
        _state_with_prefix(stage1["model_state"], "head.")
    )
    protein_std = torch.tensor(
        stage1["protein_scaler"]["scale"], dtype=torch.float32
    )
    return LongitudinalModel(
        clinical,
        protein_head,
        temporal,
        model_cfg["temporal_hidden"],
        model_cfg["clinical_hidden"],
        len(stage1["protein_names"]),
        len(stage2["target_names"]),
        model_cfg["bridge_rank"],
        protein_std,
        model_cfg["temporal_dropout"],
    ).to(device)


def _make_loader(data: Dict, indices: np.ndarray, config: Dict, shuffle: bool):
    true = data["true_delta"][indices]
    payload = {
        "true_delta": np.nan_to_num(true, nan=0.0).astype(np.float32),
        "protein_mask": np.isfinite(true),
    }
    dataset = TemporalDataset(
        data["x_pre"][indices],
        data["x_post"][indices],
        data["interval"][indices],
        payload,
    )
    return DataLoader(
        dataset,
        batch_size=config["stage3"]["batch_size"],
        shuffle=shuffle,
        num_workers=config["num_workers"],
        pin_memory=False,
    )


def _compute_losses(
    prediction,
    batch,
    protein_raw,
    protein_norm,
    target_norm,
    common_protein_indices,
    common_target_indices,
    model,
    config,
):
    predicted_delta = prediction["predicted_delta"]
    protein_loss = masked_huber(
        predicted_delta, batch["true_delta"], batch["protein_mask"]
    )
    true_path, path_mask = masked_pathway_profiles(
        batch["true_delta"], batch["protein_mask"], protein_raw
    )
    predicted_path_masked, _ = masked_pathway_profiles(
        predicted_delta, batch["protein_mask"], protein_raw
    )
    pathway_loss = masked_huber(
        predicted_path_masked, true_path, path_mask
    )
    predicted_path_full = predicted_delta.matmul(protein_norm)
    target_path = prediction["target_like"].matmul(target_norm)
    bridge_loss = masked_mse(
        target_path[:, common_target_indices],
        predicted_path_full[:, common_protein_indices],
        torch.ones_like(target_path[:, common_target_indices], dtype=torch.bool),
    )
    regularization = model.bridge.regularization(
        config["stage3"].get("bridge_l1", 0.0)
    )
    total = (
        config["stage3"]["lambda_protein"] * protein_loss
        + config["stage3"]["lambda_pathway"] * pathway_loss
        + config["stage3"]["lambda_bridge"] * bridge_loss
        + config["stage3"]["lambda_regularization"] * regularization
    )
    return total


def _fit_model(
    config: Dict,
    device,
    logger,
    stage1: Dict,
    stage2: Dict,
    data: Dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    pathways: Dict,
    checkpoint_path: Path,
):
    model = _build_model(config, stage1, stage2, device)
    train_loader = _make_loader(data, train_idx, config, True)
    val_loader = _make_loader(data, val_idx, config, False)
    groups = parameter_groups(
        [model],
        config["stage3"]["learning_rate"],
        model.clinical_encoder,
        config["stage3"]["encoder_learning_rate"],
        config["stage3"]["weight_decay"],
    )
    optimizer = AdamW(groups)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    amp_scaler = GradScaler(enabled=device.type == "cuda")
    stopping = EarlyStopping(config["stage3"]["patience"])
    protein_raw = torch.tensor(pathways["protein_raw"], device=device)
    protein_norm = torch.tensor(pathways["protein_norm"], device=device)
    target_norm = torch.tensor(pathways["target_norm"], device=device)
    common_protein = torch.tensor(
        pathways["common_protein_indices"], dtype=torch.long, device=device
    )
    common_target = torch.tensor(
        pathways["common_target_indices"], dtype=torch.long, device=device
    )
    for epoch in range(1, config["stage3"]["epochs"] + 1):
        freeze = epoch <= config["stage3"]["freeze_clinical_epochs"]
        for parameter in model.clinical_encoder.parameters():
            parameter.requires_grad = not freeze
        model.train()
        train_sum = 0.0
        train_count = 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                prediction = model(
                    batch["x_pre"], batch["x_post"], batch["interval"]
                )
                loss = _compute_losses(
                    prediction,
                    batch,
                    protein_raw,
                    protein_norm,
                    target_norm,
                    common_protein,
                    common_target,
                    model,
                    config,
                )
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            train_sum += loss.item() * batch["x_pre"].shape[0]
            train_count += batch["x_pre"].shape[0]
        model.eval()
        val_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                prediction = model(
                    batch["x_pre"], batch["x_post"], batch["interval"]
                )
                loss = _compute_losses(
                    prediction,
                    batch,
                    protein_raw,
                    protein_norm,
                    target_norm,
                    common_protein,
                    common_target,
                    model,
                    config,
                )
                val_sum += loss.item() * batch["x_pre"].shape[0]
                val_count += batch["x_pre"].shape[0]
        train_loss = train_sum / max(train_count, 1)
        val_loss = val_sum / max(val_count, 1)
        scheduler.step(val_loss)
        improved = stopping.update(val_loss)
        logger.info(
            "%s epoch %d | train %.6f | val %.6f",
            checkpoint_path.stem,
            epoch,
            train_loss,
            val_loss,
        )
        if improved:
            atomic_torch_save(
                {
                    "stage": 3,
                    "model_state": model.state_dict(),
                    "feature_names": stage1["feature_names"],
                    "protein_names": stage1["protein_names"],
                    "target_names": stage2["target_names"],
                    "clinical_scaler": stage1["clinical_scaler"],
                    "protein_scaler": stage1["protein_scaler"],
                    "model_config": config["model"],
                    "best_val_loss": val_loss,
                    "scientific_scope": (
                        "M_PT is a protein-to-target perturbation bridge. It is not "
                        "an in-vivo KD map and does not establish drug binding."
                    ),
                },
                checkpoint_path,
            )
        if stopping.should_stop:
            break
    best = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best["model_state"])
    return model


def _predict(model, data, indices, config, device):
    loader = _make_loader(data, indices, config, False)
    outputs = {"predicted_delta": [], "target_like": []}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            prediction = model(
                batch["x_pre"].to(device),
                batch["x_post"].to(device),
                batch["interval"].to(device),
            )
            for key in outputs:
                outputs[key].append(prediction[key].cpu().numpy())
    return {key: np.concatenate(value) for key, value in outputs.items()}


def _inner_split(row_indices, groups, config):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config["stage3"]["validation_fraction"],
        random_state=config["seed"],
    )
    relative_train, relative_val = next(
        splitter.split(row_indices, groups=groups[row_indices])
    )
    return row_indices[relative_train], row_indices[relative_val]


def train_stage3(
    config: Dict,
    device: torch.device,
    logger,
    stage1_checkpoint: Path,
    stage2_checkpoint: Path,
) -> Path:
    output = Path(config["output_dir"]) / "stage3"
    output.mkdir(parents=True, exist_ok=True)
    stage1 = torch.load(stage1_checkpoint, map_location="cpu")
    stage2 = torch.load(stage2_checkpoint, map_location="cpu")
    if stage1["feature_names"] != stage2["feature_names"]:
        raise ValueError("Stage 1 and Stage 2 clinical schemas differ.")
    data = _load_longitudinal(config, stage1)
    metadata = data["metadata"]
    participant_ids = np.asarray(sorted(metadata["participant_id"].unique()))
    groups = metadata["participant_id"].to_numpy()
    drug = load_drug_profiles(Path(config["data"]["root"]))
    pathways = load_pathway_matrices(
        Path(config["data"]["root"]), stage1["protein_names"], stage2["target_names"]
    )
    folds = KFold(
        n_splits=config["stage3"]["folds"],
        shuffle=True,
        random_state=config["seed"],
    )
    predictions_by_id = {}
    logger.info(
        "Stage 3: %d participants, %d imputation rows, %d folds",
        len(participant_ids),
        len(metadata),
        config["stage3"]["folds"],
    )
    for fold, (outer_train, outer_test) in enumerate(
        folds.split(participant_ids), start=1
    ):
        train_ids = set(participant_ids[outer_train])
        test_ids = set(participant_ids[outer_test])
        outer_train_rows = np.where(np.isin(groups, list(train_ids)))[0]
        test_rows = np.where(np.isin(groups, list(test_ids)))[0]
        train_rows, val_rows = _inner_split(outer_train_rows, groups, config)
        checkpoint = output / ("fold_%02d.pt" % fold)
        model = _fit_model(
            config,
            device,
            logger,
            stage1,
            stage2,
            data,
            train_rows,
            val_rows,
            pathways,
            checkpoint,
        )
        prediction = _predict(model, data, test_rows, config, device)
        test_metadata = metadata.iloc[test_rows].reset_index(drop=True)
        for participant_id in sorted(test_ids):
            selection = np.where(
                test_metadata["participant_id"].to_numpy() == participant_id
            )[0]
            predictions_by_id[participant_id] = {
                key: value[selection].mean(axis=0)
                for key, value in prediction.items()
            }
        matrix = model.bridge.matrix().detach().cpu().numpy()
        pd.DataFrame(
            matrix,
            index=stage1["protein_names"],
            columns=stage2["target_names"],
        ).to_csv(
            output / ("fold_%02d_M_PT.csv" % fold), encoding="utf-8-sig"
        )
        np.savez_compressed(
            output / ("fold_%02d_bridge_factors.npz" % fold),
            U=model.bridge.U.detach().cpu().numpy(),
            V=model.bridge.V.detach().cpu().numpy(),
        )
    ordered_ids = participant_ids.tolist()
    predicted_delta = np.stack(
        [predictions_by_id[value]["predicted_delta"] for value in ordered_ids]
    )
    target_like = np.stack(
        [predictions_by_id[value]["target_like"] for value in ordered_ids]
    )
    first_rows = (
        metadata.reset_index()
        .drop_duplicates("participant_id")
        .set_index("participant_id")
        .loc[ordered_ids]["index"]
        .to_numpy()
    )
    true_delta = data["true_delta"][first_rows]
    protein_mask = np.isfinite(true_delta)
    protein_raw = pathways["protein_raw"]
    protein_norm = pathways["protein_norm"]
    denominator = protein_mask.astype(np.float32).dot(protein_raw)
    pathway_mask = denominator > 0
    true_pathway = np.nan_to_num(true_delta, nan=0.0).dot(protein_raw) / np.maximum(
        denominator, 1.0
    )
    predicted_pathway_metric = (
        predicted_delta * protein_mask.astype(np.float32)
    ).dot(protein_raw) / np.maximum(denominator, 1.0)
    predicted_pathway = predicted_delta.dot(protein_norm)
    target_like_pathway = target_like.dot(pathways["target_norm"])
    results = output / "oof_results"
    results.mkdir(parents=True, exist_ok=True)

    def write_wide(filename, values, columns):
        frame = pd.DataFrame(values, columns=columns)
        frame.insert(0, config["data"]["id_column"], ordered_ids)
        frame.to_csv(results / filename, index=False, encoding="utf-8-sig")

    write_wide(
        "predicted_delta_protein.csv",
        predicted_delta,
        stage1["protein_names"],
    )
    write_wide("true_delta_protein.csv", true_delta, stage1["protein_names"])
    write_wide(
        "predicted_patient_pathway_profile.csv",
        predicted_pathway,
        pathways["protein_pathways"],
    )
    write_wide(
        "true_patient_pathway_profile.csv",
        true_pathway,
        pathways["protein_pathways"],
    )
    write_wide(
        "target_like_profile.csv", target_like, stage2["target_names"]
    )
    write_wide(
        "target_like_pathway_profile.csv",
        target_like_pathway,
        pathways["target_pathways"],
    )
    export_evaluation(
        results,
        ordered_ids,
        stage1["protein_names"],
        pathways["protein_pathways"],
        true_delta,
        predicted_delta,
        protein_mask,
        true_pathway,
        predicted_pathway_metric,
        pathway_mask,
    )
    patient_common = predicted_pathway[:, pathways["common_protein_indices"]]
    export_drug_similarity(
        results,
        Path(config["data"]["root"]),
        ordered_ids,
        patient_common,
        target_like,
        pathways["target_raw"],
        stage2["target_names"],
        pathways["common_target_indices"],
        pathways["common_pathways"],
        config["stage3"]["top_k_drugs"],
    )
    if config["stage3"].get("train_final_model", True):
        all_rows = np.arange(len(metadata))
        final_train, final_val = _inner_split(all_rows, groups, config)
        _fit_model(
            config,
            device,
            logger,
            stage1,
            stage2,
            data,
            final_train,
            final_val,
            pathways,
            output / "final_model.pt",
        )
    return results
