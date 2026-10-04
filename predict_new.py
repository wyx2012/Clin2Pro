import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clin2pro.config import load_config
from clin2pro.data import (
    Standardizer,
    load_pathway_matrices,
    read_clinical,
)
from clin2pro.stage3 import _build_model
from clin2pro.evaluation import export_drug_similarity
from clin2pro.utils import resolve_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict delta proteome for paired clinical CSV files."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--instance0", required=True)
    parser.add_argument("--instance2", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config["device"])
    base = Path(config["output_dir"])
    stage1 = torch.load(base / "stage1" / "best.pt", map_location="cpu")
    stage2 = torch.load(base / "stage2" / "best.pt", map_location="cpu")
    final = torch.load(base / "stage3" / "final_model.pt", map_location=device)
    model = _build_model(config, stage1, stage2, device)
    model.load_state_dict(final["model_state"])
    model.eval()
    id_column = config["data"]["id_column"]
    pre = read_clinical(Path(args.instance0), id_column)
    post = read_clinical(Path(args.instance2), id_column)
    paired = pre.merge(
        post, on=id_column, suffixes=("_pre", "_post"), validate="one_to_one"
    )
    feature_names = stage1["feature_names"]
    scaler = Standardizer.from_state_dict(stage1["clinical_scaler"])
    x_pre = scaler.transform(
        paired[[name + "_pre" for name in feature_names]].to_numpy(np.float32)
    )
    x_post = scaler.transform(
        paired[[name + "_post" for name in feature_names]].to_numpy(np.float32)
    )
    interval = (
        paired["age_at_instance_post"] - paired["age_at_instance_pre"]
    ).to_numpy(np.float32)
    if np.any(~np.isfinite(interval)) or np.any(interval <= 0):
        raise ValueError("All inferred intervals must be finite and positive.")
    with torch.no_grad():
        prediction = model(
            torch.from_numpy(x_pre).to(device),
            torch.from_numpy(x_post).to(device),
            torch.from_numpy(interval.reshape(-1, 1)).to(device),
        )
    delta = prediction["predicted_delta"].cpu().numpy()
    target_like = prediction["target_like"].cpu().numpy()
    pathways = load_pathway_matrices(
        Path(config["data"]["root"]),
        stage1["protein_names"],
        stage2["target_names"],
    )
    pathway = delta.dot(pathways["protein_norm"])
    target_pathway = target_like.dot(pathways["target_norm"])

    def write(name, values, columns):
        frame = pd.DataFrame(values, columns=columns)
        frame.insert(0, id_column, paired[id_column].astype(str).to_numpy())
        frame.to_csv(output / name, index=False, encoding="utf-8-sig")

    write("predicted_delta_protein.csv", delta, stage1["protein_names"])
    write(
        "predicted_patient_pathway_profile.csv",
        pathway,
        pathways["protein_pathways"],
    )
    write("target_like_profile.csv", target_like, stage2["target_names"])
    write(
        "target_like_pathway_profile.csv",
        target_pathway,
        pathways["target_pathways"],
    )
    export_drug_similarity(
        output,
        Path(config["data"]["root"]),
        paired[id_column].astype(str).tolist(),
        pathway[:, pathways["common_protein_indices"]],
        target_like,
        pathways["target_raw"],
        stage2["target_names"],
        pathways["common_target_indices"],
        pathways["common_pathways"],
        config["stage3"]["top_k_drugs"],
    )


if __name__ == "__main__":
    main()
