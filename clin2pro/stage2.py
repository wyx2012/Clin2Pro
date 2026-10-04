from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .data import (
    Standardizer,
    TemporalDataset,
    extract_paired_features,
    load_drug_profiles,
    load_pathway_matrices,
    mimic_path,
    pair_mimic_rows,
)
from .losses import masked_bce_logits, masked_huber, masked_mse
from .models import (
    ClinicalEncoder,
    DrugPretrainingModel,
    ProteinPredictionHead,
    TemporalPerturbationEncoder,
)
from .utils import EarlyStopping, atomic_torch_save, parameter_groups


def _load_stage1_components(config: Dict, checkpoint: Dict):
    model_cfg = config["model"]
    encoder = ClinicalEncoder(
        n_features=len(checkpoint["feature_names"]),
        hidden=model_cfg["clinical_hidden"],
        layers=model_cfg["clinical_layers"],
        heads=model_cfg["clinical_heads"],
        dropout=model_cfg["clinical_dropout"],
    )
    protein_head = ProteinPredictionHead(
        hidden=model_cfg["clinical_hidden"],
        n_proteins=len(checkpoint["protein_names"]),
        dropout=model_cfg["clinical_dropout"],
    )
    encoder_state = {
        key[len("encoder.") :]: value
        for key, value in checkpoint["model_state"].items()
        if key.startswith("encoder.")
    }
    head_state = {
        key[len("head.") :]: value
        for key, value in checkpoint["model_state"].items()
        if key.startswith("head.")
    }
    encoder.load_state_dict(encoder_state)
    protein_head.load_state_dict(head_state)
    return encoder, protein_head


def _drug_pathway_profiles(affinity, affinity_mask, membership):
    values = np.nan_to_num(affinity, nan=0.0)
    numerator = values.dot(membership)
    denominator = affinity_mask.astype(np.float32).dot(membership)
    mask = denominator > 0
    profile = numerator / np.maximum(denominator, 1.0)
    return profile.astype(np.float32), mask


def train_stage2(
    config: Dict, device: torch.device, logger, stage1_checkpoint: Path
) -> Path:
    output = Path(config["output_dir"]) / "stage2"
    checkpoint_path = output / "best.pt"
    stage1 = torch.load(stage1_checkpoint, map_location="cpu")
    feature_names = stage1["feature_names"]
    protein_names = stage1["protein_names"]
    clinical_scaler = Standardizer.from_state_dict(stage1["clinical_scaler"])
    drug = load_drug_profiles(Path(config["data"]["root"]))
    pathways = load_pathway_matrices(
        Path(config["data"]["root"]), protein_names, drug["targets"]
    )
    drug_pathway, drug_pathway_mask = _drug_pathway_profiles(
        drug["affinity"], drug["affinity_mask"], pathways["target_raw"]
    )
    drug_index = {name: i for i, name in enumerate(drug["drugs"])}
    frames = []
    for imputation in config["data"]["stage2_imputations"]:
        paired = pair_mimic_rows(
            mimic_path(Path(config["data"]["root"]), imputation),
            config["data"]["id_column"],
            config["data"].get("max_stage2_pairs_per_imputation"),
        )
        paired["_imputation"] = imputation
        frames.append(paired)
    paired = pd.concat(frames, ignore_index=True)
    x_pre, x_post = extract_paired_features(paired, feature_names)
    x_pre = clinical_scaler.transform(x_pre)
    x_post = clinical_scaler.transform(x_post)
    interval = paired["_interval_years"].to_numpy(dtype=np.float32)
    label_indices = np.asarray(
        [drug_index[name] for name in paired["drug_name"]], dtype=np.int64
    )
    payload = {
        "affinity": np.nan_to_num(drug["affinity"][label_indices], nan=0.0).astype(
            np.float32
        ),
        "affinity_mask": drug["affinity_mask"][label_indices],
        "binary": np.nan_to_num(drug["binary"][label_indices], nan=0.0).astype(
            np.float32
        ),
        "binary_mask": drug["binary_mask"][label_indices],
        "pathway": drug_pathway[label_indices],
        "pathway_mask": drug_pathway_mask[label_indices],
    }
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config["stage2"]["val_fraction"],
        random_state=config["seed"],
    )
    train_idx, val_idx = next(
        splitter.split(x_pre, groups=paired["_sample_key"].to_numpy())
    )
    train_payload = {key: value[train_idx] for key, value in payload.items()}
    val_payload = {key: value[val_idx] for key, value in payload.items()}
    train_data = TemporalDataset(
        x_pre[train_idx], x_post[train_idx], interval[train_idx], train_payload
    )
    val_data = TemporalDataset(
        x_pre[val_idx], x_post[val_idx], interval[val_idx], val_payload
    )
    train_loader = DataLoader(
        train_data,
        batch_size=config["stage2"]["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config["stage2"]["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
    )
    clinical_encoder, _ = _load_stage1_components(config, stage1)
    temporal_encoder = TemporalPerturbationEncoder(
        clinical_hidden=config["model"]["clinical_hidden"],
        n_features=len(feature_names),
        hidden=config["model"]["temporal_hidden"],
        dropout=config["model"]["temporal_dropout"],
    )
    model = DrugPretrainingModel(
        clinical_encoder,
        temporal_encoder,
        config["model"]["temporal_hidden"],
        len(drug["targets"]),
        len(pathways["target_pathways"]),
    ).to(device)
    groups = parameter_groups(
        [model],
        config["stage2"]["learning_rate"],
        model.clinical_encoder,
        config["stage2"]["encoder_learning_rate"],
        config["stage2"]["weight_decay"],
    )
    optimizer = AdamW(groups)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    amp_scaler = GradScaler(enabled=device.type == "cuda")
    stopping = EarlyStopping(config["stage2"]["patience"])
    logger.info(
        "Stage 2: %d augmented pairs (%d unique episodes), %d targets, %d pathways",
        len(paired),
        paired["_sample_key"].nunique(),
        len(drug["targets"]),
        len(pathways["target_pathways"]),
    )

    def compute_loss(batch, prediction):
        affinity_loss = masked_huber(
            prediction["affinity"], batch["affinity"], batch["affinity_mask"]
        )
        binary_loss = masked_bce_logits(
            prediction["binary_logits"], batch["binary"], batch["binary_mask"]
        )
        pathway_loss = masked_mse(
            prediction["pathway"], batch["pathway"], batch["pathway_mask"]
        )
        total = (
            config["stage2"]["lambda_affinity"] * affinity_loss
            + config["stage2"]["lambda_binary"] * binary_loss
            + config["stage2"]["lambda_pathway"] * pathway_loss
        )
        return total

    for epoch in range(1, config["stage2"]["epochs"] + 1):
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
                loss = compute_loss(batch, prediction)
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
                loss = compute_loss(batch, prediction)
                val_sum += loss.item() * batch["x_pre"].shape[0]
                val_count += batch["x_pre"].shape[0]
        train_loss = train_sum / max(train_count, 1)
        val_loss = val_sum / max(val_count, 1)
        scheduler.step(val_loss)
        improved = stopping.update(val_loss)
        logger.info(
            "Stage 2 epoch %d | train %.6f | val %.6f",
            epoch,
            train_loss,
            val_loss,
        )
        if improved:
            atomic_torch_save(
                {
                    "stage": 2,
                    "model_state": model.state_dict(),
                    "feature_names": feature_names,
                    "protein_names": protein_names,
                    "target_names": drug["targets"],
                    "drug_names": drug["drugs"],
                    "target_pathways": pathways["target_pathways"],
                    "clinical_scaler": stage1["clinical_scaler"],
                    "protein_scaler": stage1["protein_scaler"],
                    "model_config": config["model"],
                    "best_val_loss": val_loss,
                },
                checkpoint_path,
            )
        if stopping.should_stop:
            break
    return checkpoint_path
