# Model and evaluation

## Three-stage model

| Stage | Input | Supervision / output |
| --- | --- | --- |
| 1: cross-sectional pretraining | One clinical assessment | Contemporaneous protein abundance |
| 2: temporal pretraining | Before/after clinical states and elapsed time around drug exposure | Drug-target affinity, binary interaction and pathway profiles |
| 3: longitudinal prediction | Paired clinical assessments and elapsed time | Post-minus-pre protein changes, pathway profiles and target-like scores |

Stage 1 uses a numerical-feature tokenizer, Transformer clinical encoder and protein prediction head. Stage 2 initializes the clinical encoder from Stage 1 and learns a temporal perturbation encoder. Stage 3 initializes its clinical/temporal encoders from Stage 2 and its protein head from Stage 1.

The longitudinal prediction combines a cross-sectional difference with a learned temporal residual:

```text
predicted_delta = baseline_delta + temporal_residual
baseline_delta = (protein_head(z_post) - protein_head(z_pre)) * protein_std
target_like = predicted_delta @ U @ V.T
```

Protein changes are projected through fixed pathway membership matrices. The trainable bridge `M_PT = U @ V.T` is low rank. Stage 3 combines protein, pathway and bridge-consistency losses with regularization; drug-pattern similarities are computed after training.

## Configurations

The supplied full configuration uses seed 2026, a clinical hidden size of 256, four Transformer layers, temporal hidden size 384, bridge rank 32 and five Stage 3 outer folds. Stage 1 uses LabTOP; Stages 2 and 3 use LabTOP, MICE, SVD and MedianMode inputs. The study-specific schema describes 74 clinical features, 1,463 proteins and 65 targets, with 146 longitudinal participants. The actual model dimensions and column order come from the data/checkpoints.

`configs/debug.yaml` limits training data/epochs and uses smaller models. It checks the workflow with supplied data; it is not a data-free demonstration or a substitute for full evaluation. Keep different experiment outputs in separate directories by changing `output_dir`.

## Splits and missing data

- Stage 1 fits standardization on its training split and is configured to exclude longitudinal participant IDs. Separate OOF export retrains fold-specific models.
- Stage 2 groups imputed copies by exposure episode. Different episodes from the same patient may appear in different splits.
- Stage 3 splits by participant across outer folds and uses an inner validation split for early stopping. Imputed copies remain together within each split.
- Missing protein and drug-target labels are masked. Pathway supervision uses the observable member proteins for each sample.

The pipeline consumes pre-imputed inputs. Upstream imputation fitting and leakage controls must be documented separately; downstream group splits alone do not establish leakage-free preprocessing.

## Evaluation entry points

Run these from the repository root with the same configuration used for training. All support `--help`.

| Script in `scripts/` | Required prior work | Result |
| --- | --- | --- |
| `export_stage1_predictions.py` | Stage 1 training and cross-sectional inputs | Fitted/validation exports plus OOF retraining unless `--skip-oof` |
| `export_stage2_results.py` | Stages 1–2 and exposure inputs | Affinity, interaction, pathway and embedding exports |
| `export_stage1_longitudinal_endpoints.py` | Stage 1 and paired longitudinal inputs | Estimated protein abundance at both endpoints and their difference |
| `export_stage3_benchmarks.py` | Full training and Stage 3 OOF outputs | Zero change, training mean, Stage 1 difference, ridge, elastic net and plain MLP comparisons |

The baseline comparison reuses the full model's OOF predictions and trains comparison models using the corresponding participant folds. It does not retrain the full Stage 3 model.

## Output files

`outputs/full/stage3/oof_results/` contains:

| Files | Meaning |
| --- | --- |
| `predicted_delta_protein.csv`, `true_delta_protein.csv` | Predicted and measured post-minus-pre protein changes |
| `predicted_patient_pathway_profile.csv`, `true_patient_pathway_profile.csv` | Protein-derived pathway profiles |
| `target_like_profile.csv`, `target_like_pathway_profile.csv` | Bridge outputs and their pathway projections |
| `drug_similarity_all.csv`, `drug_similarity_top20.csv` | Exploratory drug-pattern similarities |
| `metrics_summary.json`, `metrics_per_protein.csv`, `metrics_per_pathway.csv` | Aggregate and per-output metrics |

Each outer fold also saves `fold_XX.pt`, `fold_XX_M_PT.csv` and `fold_XX_bridge_factors.npz`. The full configuration enables final-model training. Use held-out/OOF predictions to assess performance, and `stage3/final_model.pt` for subsequent inference.

## Verification scope

The existing core tests cover protein-name normalization, standardization with missing/constant values, masked loss, pathway denominators and low-rank bridge factorization. The command-line entry points can be checked with `--help` without clinical data. These checks do not reproduce complete training runs or independently establish manuscript performance.

Only core training, inference and evaluation are included here. Cohort-specific external analyses, manuscript plotting and upstream knowledge-table curation are outside this repository's scope.
