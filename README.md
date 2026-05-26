# M2M-ALK

Reference implementation for **M2M-ALK**, a patient-level multimodal model for prioritising confirmatory ALK rearrangement testing in lung adenocarcinoma using CT radiological features and H&E whole-slide image (WSI) pathological features.

This repository contains cleaned, configurable scripts derived from the research code. Personal file paths, duplicated model definitions, and manuscript-inconsistent defaults have been removed.

## What is included

```text
m2m_alk/
  ct_model.py          # 3D ResNet18 CT stream with 512-dim penultimate features
  data.py              # CT ROI and patient-level multimodal datasets
  fusion_models.py     # Concatenation, gated fusion, and M2M-ALK attention fusion
  utils.py             # metrics, I/O, checkpoints, reproducibility helpers
scripts/
  train_ct.py          # train CT-only model
  extract_ct_features.py
  train_fusion.py      # train concat/gated/attention fusion models
  gradcam_ct.py        # CT Grad-CAM visualisation
  shap_interpret.py    # modality and feature-level SHAP interpretation
```

## Data expected by the scripts

This repository intentionally does **not** include reference data, demo patient records, trained weights, WSI feature tensors, CT ROI volumes, or example CSV/XLSX files.

### CT ROI metadata

`train_ct.py`, `extract_ct_features.py`, and `gradcam_ct.py` expect a user-provided CSV/XLSX metadata table. By default, column 0 is the sample identifier and column 1 is the binary label. CT ROI files are resolved as:

```text
<roi-root>/<split-name>/<sample_id>.nii.gz
```

Input volumes should already be cropped/resampled to `32 x 32 x 32` voxels. The script applies CT intensity clipping/scaling, nonzero channel-wise normalisation, and the training augmentations described in the supplementary methods.

### Multimodal fusion table

`train_fusion.py` and `shap_interpret.py` expect user-provided patient-level feature tables. By default, column 0 is the sample identifier, column 1 is the path to a 512-dimensional WSI feature tensor, column 2 is the binary label, and columns 3 onward are CT features. CT features are standardised with a `StandardScaler` fitted only on the training set.

## Main commands

Train the CT model:

```bash
python scripts/train_ct.py \
  --roi-root /path/to/32resample_roi \
  --train-metadata data/training.xlsx \
  --val-metadata data/val.xlsx \
  --external-metadata data/external.xlsx \
  --pretrained-weights /path/to/resnet_18_23dataset.pth \
  --output-dir outputs/ct_resnet18
```

Extract 512-dimensional CT features:

```bash
python scripts/extract_ct_features.py \
  --roi-root /path/to/32resample_roi \
  --metadata data/val.xlsx \
  --split-name val \
  --checkpoint outputs/ct_resnet18/best_ct_resnet18.pth \
  --output-csv outputs/ct_val_features.csv
```

Train M2M-ALK attention fusion:

```bash
python scripts/train_fusion.py \
  --model attention \
  --train-csv data/fusion_train.csv \
  --val-csv data/fusion_val.csv \
  --external-csv data/fusion_external.csv \
  --output-dir outputs/m2m_alk
```

Generate CT Grad-CAM overlays:

```bash
python scripts/gradcam_ct.py \
  --roi-root /path/to/32resample_roi \
  --metadata data/val.xlsx \
  --split-name val \
  --checkpoint outputs/ct_resnet18/best_ct_resnet18.pth \
  --output-dir outputs/gradcam
```

Run SHAP interpretation:

```bash
python scripts/shap_interpret.py \
  --train-csv data/fusion_train.csv \
  --eval-csv data/fusion_val.csv \
  --checkpoint outputs/m2m_alk/best_attention.pth \
  --output-dir outputs/shap
```


## Acknowledgements

This project builds on and compares with open computational pathology resources. We thank the authors and maintainers of the following projects/models for making their work available to the research community. This repository does **not** redistribute third-party model weights; please obtain each model from its official source and follow its license, citation, and access requirements.

| Resource | Official link | How it relates to this repository |
| --- | --- | --- |
| CLAM | https://github.com/mahmoodlab/CLAM | CLAM provides the weakly supervised multiple-instance learning framework used for patient-level WSI representation learning and attention heatmap visualisation. |
| CTransPath | https://github.com/Xiyue-Wang/TransPath | Pathology foundation model benchmarked as a WSI encoder. |
| CONCH v1.5 | https://huggingface.co/MahmoodLab/conchv1_5 | Pathology vision-language foundation model benchmarked as a WSI encoder. |
| UNI2 | https://huggingface.co/MahmoodLab/UNI2-h | Pathology foundation model benchmarked as a WSI encoder. |
| Prov-GigaPath | https://github.com/prov-gigapath/prov-gigapath | Whole-slide pathology foundation model benchmarked as a WSI encoder. |
| Virchow2 | https://huggingface.co/paige-ai/Virchow2 | Pathology foundation model used as the final fixed WSI feature extractor in M2M-ALK. |

## Installation

```bash
pip install -r requirements.txt
```

The code was structured for Python 3.10+.

## Notes for open release

This repository does **not** include patient data, trained weights, or identifiable file paths. Before publishing, add an appropriate license and data-access statement for your institution/journal policy.
