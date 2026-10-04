"""Export Stage 1 pre/post proteome predictions for the longitudinal cohort.

The Stage 3 benchmark already exports f(post) - f(pre), but not the two
absolute Stage 1 predictions needed for a midpoint-aligned comparison plot.
This script uses the frozen Stage 1 checkpoint and averages predictions across
the same four clinical-imputation versions used by Stage 3.

Run from the project root:
    python scripts/export_stage1_longitudinal_endpoints.py --config configs/full.yaml
"""

import argparse
import sys
from pathlib import Path

# Support direct execution from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from clin2pro.config import load_config
from clin2pro.models import ClinicalEncoder, ProteinPredictionHead
from clin2pro.stage3 import _load_longitudinal
from clin2pro.utils import resolve_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export longitudinal Stage 1 pre/post proteome predictions."
    )
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Default: <output_dir>/stage1/best.pt",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Default: <output_dir>/stage3/stage1_longitudinal_endpoints",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_model(stage1, config, device):
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
    clinical.load_state_dict(
        {
            key[len("encoder.") :]: value
            for key, value in stage1["model_state"].items()
            if key.startswith("encoder.")
        }
    )
    head.load_state_dict(
        {
            key[len("head.") :]: value
            for key, value in stage1["model_state"].items()
            if key.startswith("head.")
        }
    )
    return clinical.to(device).eval(), head.to(device).eval()


def predict(clinical, head, values, mean, scale, batch_size, device):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).to(device)
            standardized = head(clinical(batch)).cpu().numpy()
            chunks.append(standardized * scale + mean)
    return np.concatenate(chunks, axis=0).astype(np.float32)


def aggregate(values, participant_ids, ordered_ids):
    frame = pd.DataFrame(values)
    frame.insert(0, "participant_id", participant_ids)
    grouped = frame.groupby("participant_id", sort=False).mean()
    return grouped.loc[ordered_ids].to_numpy(dtype=np.float32)


def write_wide(path, values, ordered_ids, id_column, protein_names):
    frame = pd.DataFrame(values, columns=protein_names)
    frame.insert(0, id_column, ordered_ids)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    args = parse_args()
    config = load_config(args.config)
    project_root = Path(args.config).resolve().parent.parent
    output_root = (project_root / config["output_dir"]).resolve()
    checkpoint = (
        Path(args.checkpoint).resolve()
        if args.checkpoint
        else output_root / "stage1" / "best.pt"
    )
    output_dir = (
        Path(args.output).resolve()
        if args.output
        else output_root / "stage3" / "stage1_longitudinal_endpoints"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(config["device"])
    stage1 = torch_load(checkpoint, map_location="cpu")
    data = _load_longitudinal(config, stage1)
    metadata = data["metadata"].copy()
    participant_ids = metadata["participant_id"].astype(str).to_numpy()
    ordered_ids = sorted(metadata["participant_id"].astype(str).unique())

    clinical, head = build_model(stage1, config, device)
    protein_mean = np.asarray(
        stage1["protein_scaler"]["mean"], dtype=np.float32
    ).reshape(1, -1)
    protein_scale = np.asarray(
        stage1["protein_scaler"]["scale"], dtype=np.float32
    ).reshape(1, -1)
    pre_rows = predict(
        clinical,
        head,
        data["x_pre"],
        protein_mean,
        protein_scale,
        args.batch_size,
        device,
    )
    post_rows = predict(
        clinical,
        head,
        data["x_post"],
        protein_mean,
        protein_scale,
        args.batch_size,
        device,
    )
    pre = aggregate(pre_rows, participant_ids, ordered_ids)
    post = aggregate(post_rows, participant_ids, ordered_ids)
    delta = post - pre

    id_column = config["data"]["id_column"]
    protein_names = stage1["protein_names"]
    write_wide(
        output_dir / "stage1_predicted_proteome_instance0.csv",
        pre,
        ordered_ids,
        id_column,
        protein_names,
    )
    write_wide(
        output_dir / "stage1_predicted_proteome_instance2.csv",
        post,
        ordered_ids,
        id_column,
        protein_names,
    )
    write_wide(
        output_dir / "stage1_predicted_delta_from_endpoints.csv",
        delta,
        ordered_ids,
        id_column,
        protein_names,
    )
    print(
        "Exported Stage 1 endpoints for %d participants and %d proteins to %s"
        % (len(ordered_ids), len(protein_names), output_dir)
    )


if __name__ == "__main__":
    main()
