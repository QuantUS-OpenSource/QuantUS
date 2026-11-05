# MedSAM2 checkpoints for inference

Weights used by the MedSAM2 inference notebooks in `engines/ceus/CLI-Demos/`.
Checkpoints are **not tracked by git** (`*.pt` is git-ignored; each is ~470 MB).

```
AI_inference/MedSAM2/checkpoints/
    MedSAM2_MRI_LiverLesion.pt   # default for the inference notebooks (pretrained MedSAM2 liver lesion model)
```

## Cross-validation checkpoints (one per fold)

The model was fine-tuned with 5-fold, leave-patients-out cross-validation. Put the
checkpoint chosen for each fold here as:

```
checkpoints/fold_1.pt  fold_2.pt  fold_3.pt  fold_4.pt  fold_5.pt
```

The inference notebooks then segment each patient with the checkpoint of the fold that
**held that patient out**, so no patient is ever segmented by a model that trained on it
(`engines/ceus/CLI-Demos/medsam2_function/fold_checkpoints.py`).

`fold_assignment.json` (patient -> fold) is generated from the training repo's
`manifest.csv` and `kfold_split.json`. It contains patient ids, so it is git-ignored:

```bash
python engines/ceus/CLI-Demos/medsam2_function/fold_checkpoints.py \
    --manifest <MedSAM2_finetune_data>/manifest.csv \
    --kfold-split <MedSAM2_finetune_data>/kfold/kfold_split.json
```

Fine-tuning checkpoints include optimizer state (~470 MB). Inference only needs the
`{'model': state_dict}` part, so a stripped copy (~155 MB) works as a drop-in.

## Other checkpoints / new patients

Patients outside the cross-validation set have no held-out model. For them set the
`MEDSAM2_CHECKPOINT` environment variable (full path) to force one checkpoint for every
case, e.g. the pretrained `checkpoints/MedSAM2_MRI_LiverLesion.pt` or a model trained on
all patients. Set `MEDSAM2_FOLD_DIR` to use a different folder than this one.

## Setup

Install the GPU dependencies (includes the `sam2` package):

```bash
pip install -r requirements.txt -r requirements_GPU.txt
```

Training and evaluation live in the separate **3DMPUS-segmentation** repository.
