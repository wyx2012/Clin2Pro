import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


IMPUTATION_FILES = {
    "LabTOP": "LabTOP填补",
    "MICE": "MICE迭代填补",
    "SVD": "低秩SVD填补",
    "MedianMode": "中位数众数填补",
}


def canonical_feature_name(name: str) -> str:
    name = str(name).strip().replace("\ufeff", "")
    return re.sub(r"_(?:i0|i2)__", "__", name)


def canonical_protein_name(name: str) -> str:
    return str(name).strip().strip('"').split(";", 1)[0].strip()


def _clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [str(c).strip().replace("\ufeff", "") for c in frame.columns]
    return frame


def read_clinical(path: Path, id_column: str) -> pd.DataFrame:
    frame = _clean_columns(pd.read_csv(path))
    frame = frame.rename(columns={c: canonical_feature_name(c) for c in frame.columns})
    frame[id_column] = frame[id_column].astype(str)
    non_numeric = {id_column, "drug_name", "measurement_phase"}
    for column in frame.columns:
        if column not in non_numeric:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def read_olink(
    path: Path, id_column: str, nrows: Optional[int] = None
) -> pd.DataFrame:
    header = _clean_columns(pd.read_csv(path, nrows=0))
    dtype = {column: "float32" for column in header.columns if column != id_column}
    frame = pd.read_csv(path, nrows=nrows, dtype=dtype)
    frame = _clean_columns(frame)
    rename = {
        column: canonical_protein_name(column)
        for column in frame.columns
        if column != id_column
    }
    frame = frame.rename(columns=rename)
    if len(set(rename.values())) != len(rename):
        raise ValueError("Canonical Olink protein names are not unique.")
    frame[id_column] = frame[id_column].astype(str)
    return frame


class Standardizer:
    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        mean = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0)
        mean = np.nan_to_num(mean, nan=0.0)
        scale = np.nan_to_num(scale, nan=1.0)
        scale[scale < 1e-6] = 1.0
        return cls(mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        output = (values - self.mean) / self.scale
        return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean

    def state_dict(self) -> Dict[str, List[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_state_dict(cls, state: Dict[str, List[float]]) -> "Standardizer":
        return cls(np.asarray(state["mean"]), np.asarray(state["scale"]))


class CrossSectionalDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(np.nan_to_num(y, nan=0.0).astype(np.float32))
        self.mask = torch.from_numpy(mask.astype(np.bool_))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int):
        return self.x[index], self.y[index], self.mask[index]


class TemporalDataset(Dataset):
    def __init__(
        self,
        x_pre: np.ndarray,
        x_post: np.ndarray,
        interval_years: np.ndarray,
        payload: Dict[str, np.ndarray],
    ) -> None:
        self.x_pre = torch.from_numpy(x_pre.astype(np.float32))
        self.x_post = torch.from_numpy(x_post.astype(np.float32))
        self.interval = torch.from_numpy(interval_years.astype(np.float32)).reshape(-1, 1)
        self.payload = {
            key: torch.from_numpy(value)
            for key, value in payload.items()
        }

    def __len__(self) -> int:
        return self.x_pre.shape[0]

    def __getitem__(self, index: int):
        item = {
            "x_pre": self.x_pre[index],
            "x_post": self.x_post[index],
            "interval": self.interval[index],
        }
        item.update({key: value[index] for key, value in self.payload.items()})
        return item


def stage1_paths(root: Path, imputation: str) -> Tuple[Path, Path]:
    suffix = IMPUTATION_FILES[imputation]
    folder = root / "横截面数据（用于预训练）"
    return (
        folder / ("横截面预训练_临床结构化_%s.csv" % suffix),
        folder / "横截面预训练_Olink蛋白组.csv",
    )


def longitudinal_paths(
    root: Path, imputation: str
) -> Tuple[Path, Path, Path, Path]:
    suffix = IMPUTATION_FILES[imputation]
    folder = root / "两次记录146人"
    return (
        folder / ("临床结构化_Instance0_%s.csv" % suffix),
        folder / ("临床结构化_Instance2_%s.csv" % suffix),
        folder / "Olink蛋白组_Instance0.csv",
        folder / "Olink蛋白组_Instance2.csv",
    )


def mimic_path(root: Path, imputation: str) -> Path:
    suffix = IMPUTATION_FILES[imputation]
    return (
        root
        / "MIMIC用药前后记录"
        / ("MIMIC数据_%s_BindingDB_Davis.csv" % suffix)
    )


def pair_mimic_rows(
    path: Path,
    id_column: str,
    max_pairs: Optional[int] = None,
) -> pd.DataFrame:
    frame = read_clinical(path, id_column)
    frame["measurement_phase"] = (
        frame["measurement_phase"].astype(str).str.lower().str.strip()
    )
    frame["_episode"] = frame.groupby(
        [id_column, "drug_name", "measurement_phase"], sort=False
    ).cumcount()
    keys = [id_column, "drug_name", "_episode"]
    pre = frame[frame["measurement_phase"].eq("pre")].drop(
        columns=["measurement_phase"]
    )
    post = frame[frame["measurement_phase"].eq("post")].drop(
        columns=["measurement_phase"]
    )
    paired = pre.merge(post, on=keys, suffixes=("_pre", "_post"), validate="one_to_one")
    paired = paired.copy()
    paired["_interval_years"] = (
        paired["age_at_instance_post"] - paired["age_at_instance_pre"]
    )
    paired = paired[
        np.isfinite(paired["_interval_years"]) & (paired["_interval_years"] > 0)
    ].copy()
    paired["_sample_key"] = (
        paired[id_column].astype(str)
        + "|"
        + paired["drug_name"].astype(str)
        + "|"
        + paired["_episode"].astype(str)
    )
    if max_pairs is not None:
        paired = paired.iloc[: int(max_pairs)].copy()
    return paired


def extract_paired_features(
    paired: pd.DataFrame,
    feature_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    pre_columns = [name + "_pre" for name in feature_names]
    post_columns = [name + "_post" for name in feature_names]
    missing = [
        column
        for column in pre_columns + post_columns
        if column not in paired.columns
    ]
    if missing:
        raise ValueError("Missing paired clinical columns: %s" % missing[:10])
    return (
        paired[pre_columns].to_numpy(dtype=np.float32),
        paired[post_columns].to_numpy(dtype=np.float32),
    )


def load_pathway_matrices(
    root: Path, protein_names: Sequence[str], target_names: Sequence[str]
):
    folder = root / "通路数据"
    protein = pd.read_csv(folder / "血浆蛋白-通路矩阵.csv", index_col=0)
    target = pd.read_csv(folder / "药物靶点-通路矩阵.csv", index_col=0)
    protein.index = protein.index.map(canonical_protein_name)
    target.index = target.index.map(canonical_protein_name)
    missing_proteins = sorted(set(protein_names) - set(protein.index))
    missing_targets = sorted(set(target_names) - set(target.index))
    if missing_proteins:
        raise ValueError(
            "Proteins absent from protein-pathway matrix: %s"
            % missing_proteins[:10]
        )
    if missing_targets:
        raise ValueError(
            "Targets absent from target-pathway matrix: %s" % missing_targets[:10]
        )
    protein = protein.reindex(list(protein_names)).fillna(0.0)
    target = target.reindex(list(target_names)).fillna(0.0)
    if protein.index.has_duplicates or target.index.has_duplicates:
        raise ValueError("Duplicate canonical proteins in pathway matrices.")
    common = [column for column in protein.columns if column in set(target.columns)]
    protein_raw = protein.to_numpy(dtype=np.float32)
    target_raw = target.to_numpy(dtype=np.float32)
    protein_norm = protein_raw / np.maximum(protein_raw.sum(axis=0), 1.0)
    target_norm = target_raw / np.maximum(target_raw.sum(axis=0), 1.0)
    common_protein_indices = np.asarray(
        [protein.columns.get_loc(column) for column in common], dtype=np.int64
    )
    common_target_indices = np.asarray(
        [target.columns.get_loc(column) for column in common], dtype=np.int64
    )
    return {
        "protein_raw": protein_raw,
        "protein_norm": protein_norm,
        "protein_pathways": list(protein.columns),
        "target_raw": target_raw,
        "target_norm": target_norm,
        "target_pathways": list(target.columns),
        "common_pathways": common,
        "common_protein_indices": common_protein_indices,
        "common_target_indices": common_target_indices,
    }


def load_drug_profiles(root: Path):
    drug_folder = root / "药靶"
    regression = pd.read_csv(drug_folder / "BindingDB_Davis.csv")
    classification = pd.read_csv(
        drug_folder / "BindingDB_Davis_classification.csv"
    )
    for frame in (regression, classification):
        frame["drug_name"] = frame["drug_name"].astype(str).str.strip()
        frame["target"] = frame["target_protein_name"].map(canonical_protein_name)
    drugs = sorted(set(regression["drug_name"]) | set(classification["drug_name"]))
    targets = sorted(set(regression["target"]) | set(classification["target"]))
    affinity = regression.pivot_table(
        index="drug_name", columns="target", values="affinity", aggfunc="mean"
    ).reindex(index=drugs, columns=targets)
    binary = classification.pivot_table(
        index="drug_name", columns="target", values="label", aggfunc="max"
    ).reindex(index=drugs, columns=targets)
    return {
        "drugs": drugs,
        "targets": targets,
        "affinity": affinity.to_numpy(dtype=np.float32),
        "affinity_mask": affinity.notna().to_numpy(dtype=np.bool_),
        "binary": binary.to_numpy(dtype=np.float32),
        "binary_mask": binary.notna().to_numpy(dtype=np.bool_),
    }
