# Input data specification

No participant rows are included in this release. Directory and column names below describe the existing loader contract. CSVs use a header row; UTF-8 is recommended. Keep units, coding and preprocessing consistent with the training data. This repository consumes pre-imputed clinical files; it does not implement the four imputation algorithms.

## Required filenames under data.root

For each imputation method, substitute the suffix:

| Configuration value | Filename suffix |
| --- | --- |
| `LabTOP` | `LabTOP填补` |
| `MICE` | `MICE迭代填补` |
| `SVD` | `低秩SVD填补` |
| `MedianMode` | `中位数众数填补` |

| Directory | Required files |
| --- | --- |
| `横截面数据（用于预训练）` | `横截面预训练_临床结构化_<suffix>.csv`; `横截面预训练_Olink蛋白组.csv` |
| `MIMIC用药前后记录` | `MIMIC数据_<suffix>_BindingDB_Davis.csv` |
| `两次记录146人` | `临床结构化_Instance0_<suffix>.csv`; `临床结构化_Instance2_<suffix>.csv`; `Olink蛋白组_Instance0.csv`; `Olink蛋白组_Instance2.csv` |
| `通路数据` | `血浆蛋白-通路矩阵.csv`; `药物靶点-通路矩阵.csv` |
| `药靶` | `BindingDB_Davis.csv`; `BindingDB_Davis_classification.csv` |

Supply each imputation selected in the YAML, not just one arbitrarily chosen version.

With the default `exclude_longitudinal_participants_from_stage1: true`, Stage 1 training and its exports also read `两次记录146人/临床结构化_Instance0_LabTOP填补.csv` to exclude longitudinal participant IDs. This file is needed even when running only `--stages 1`.

## Table schemas

- Clinical tables: `Participant ID` plus numeric clinical/demographic columns. `_i0__` and `_i2__` are normalized to `__`. The inferred interval is post minus pre `age_at_instance` in years.
- MIMIC tables additionally need `drug_name` and `measurement_phase` (`pre`/`post`). Repeated rows are paired in source order within participant, drug and phase. Nonpositive or nonfinite intervals are excluded.
- Olink tables: `Participant ID` followed by numeric protein columns. Protein names are normalized to the segment before `;`. Missing protein labels are masked in the objective.
- Affinity table: `drug_name`, `target_protein_name`, `affinity`. Values are consumed as supplied; the code does not convert raw KD to pKD. Supply the intended pKD-like scale.
- Classification table: `drug_name`, `target_protein_name`, `label` (0/1). Unobserved combinations are masked.
- Pathway matrices: first column is the protein/target row identifier; other columns are pathway names and binary memberships. All model proteins/targets must be represented. Shared pathways are aligned by name.

The clinical feature list below is taken from the source CSV **header only**, normalized exactly as in the loader; no rows, statistics or identifiers were copied. Units and encodings still need an author-provided dictionary.

## Normalized clinical header (74 columns)

```text
pop__Sex
cbc__Basophill count
cbc__Basophill percentage
cbc__Eosinophill count
cbc__Eosinophill percentage
cbc__Haematocrit percentage
cbc__Haemoglobin concentration
cbc__High light scatter reticulocyte count
cbc__High light scatter reticulocyte percentage
cbc__Immature reticulocyte fraction
cbc__Lymphocyte count
cbc__Lymphocyte percentage
cbc__Mean corpuscular haemoglobin
cbc__Mean corpuscular haemoglobin concentration
cbc__Mean corpuscular volume
cbc__Mean platelet (thrombocyte) volume
cbc__Mean reticulocyte volume
cbc__Mean sphered cell volume
cbc__Monocyte count
cbc__Monocyte percentage
cbc__Neutrophill count
cbc__Neutrophill percentage
cbc__Nucleated red blood cell count
cbc__Nucleated red blood cell percentage
cbc__Platelet count
cbc__Platelet crit
cbc__Platelet distribution width
cbc__Red blood cell (erythrocyte) count
cbc__Red blood cell (erythrocyte) distribution width
cbc__Reticulocyte count
cbc__Reticulocyte percentage
cbc__White blood cell (leukocyte) count
biochem__Alanine aminotransferase
biochem__Albumin
biochem__Alkaline phosphatase
biochem__Apolipoprotein A
biochem__Apolipoprotein B
biochem__Aspartate aminotransferase
biochem__C-reactive protein
biochem__Calcium
biochem__Cholesterol
biochem__Creatinine
biochem__Cystatin C
biochem__Direct bilirubin
biochem__Gamma glutamyltransferase
biochem__Glucose
biochem__Glycated haemoglobin (HbA1c)
biochem__HDL cholesterol
biochem__IGF-1
biochem__LDL direct
biochem__Lipoprotein A
biochem__Oestradiol
biochem__Phosphate
biochem__Rheumatoid factor
biochem__SHBG
biochem__Testosterone
biochem__Total bilirubin
biochem__Total protein
biochem__Triglycerides
biochem__Urate
biochem__Urea
biochem__Vitamin D
long__has_diag_asthma_copd
long__has_diag_diabetes
long__has_diag_hypertension
long__has_diag_ckd_renal
long__has_diag_cvd_ihd_mi
long__has_diag_stroke_cerebrovascular
long__has_diag_neurodegenerative_dementia
long__has_diag_cancer_any
long__has_diag_autoimmune_inflammatory
long__has_diag_liver_gi
long__has_diag_mental_health
age_at_instance
```
