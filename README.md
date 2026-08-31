# Pathology Cell-Type Fusion

Minimal research code for predicting broad transcriptomics-derived cell labels from cell-centred H&E images. The model separates three sources of evidence:

1. the raw H&E field around a target cell;
2. a target-focused version of the same field;
3. the appearance, morphology and spatial arrangement of neighbouring cells.

The two image views are encoded by a shared Phikon-v2 backbone and combined by a feature-wise gate. A target-feature-excluded graph transformer models the 8, 24 and 64 nearest neighbours. The final model freezes both branches and adds graph information to the image representation through a zero-initialised gated residual.

## Repository structure

```text
src/
  data.py                       datasets and multi-scale neighbourhood construction
  models.py                     image, graph and hierarchical fusion models
  precompute_graph_features.py  masked-patch Phikon-v2 embedding cache
  train.py                      training entry point for all three stages
requirements.txt
```

Original images, prepared datasets, checkpoints and experiment outputs are intentionally excluded.

## Data format

Place a prepared dataset outside the repository or under `data/`. It must contain `metadata.csv`, `label_mapping_coarse.json`, and the image paths referenced by the metadata.

Required metadata columns:

```text
sample_id, cell_id, split, coarse_label_index,
x_hires, y_hires,
raw_patch_path, masked_target_patch_path,
mask_area, attention_mask_area, mask_bbox_width,
mask_bbox_height, fullres_polygon_area
```

`split` must contain `train`, `val` and `test`. Image paths are relative to the dataset directory.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p checkpoints outputs
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

## Training

Run commands from the repository root.

Train the dual-view image model:

```bash
python src/train.py image \
  --data-dir data/clean_data \
  --output outputs/image \
  --batch-size 16
```

Precompute the masked-patch features used by the graph:

```bash
python src/precompute_graph_features.py \
  --data-dir data/clean_data \
  --image-checkpoint outputs/image/best_model.pth \
  --output checkpoints/graph_features.pt
```

Train the neighbour-only multi-scale graph:

```bash
python src/train.py graph \
  --data-dir data/clean_data \
  --graph-cache checkpoints/graph_features.pt \
  --output outputs/graph \
  --batch-size 64 \
  --lr 1e-4
```

Train the hierarchical image-graph fusion head:

```bash
python src/train.py fusion \
  --data-dir data/clean_data \
  --graph-cache checkpoints/graph_features.pt \
  --image-checkpoint outputs/image/best_model.pth \
  --graph-checkpoint outputs/graph/best_model.pth \
  --output outputs/fusion
```

Each stage writes its best validation checkpoint, training history, test metrics and test probabilities to the selected output directory. Macro-F1 is used for checkpoint selection.

## Scope

This folder contains the final modelling path only. Dataset-specific label cleaning, exploratory baselines, ablations, plotting scripts, dissertation assets, cluster job files and generated results are not part of the release.

