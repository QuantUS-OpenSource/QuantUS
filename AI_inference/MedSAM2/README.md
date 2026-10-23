# MedSAM2 checkpoints for inference

Weights used by the MedSAM2 inference notebooks in `engines/ceus/CLI-Demos/`.
Checkpoints are **not tracked by git** (`*.pt` is git-ignored; each is ~470 MB).

```
AI_inference/MedSAM2/checkpoints/
    MedSAM2_MRI_LiverLesion.pt   # default for the inference notebooks (pretrained MedSAM2 liver lesion model)
```

## Using a different checkpoint

Drop another `.pt` into `checkpoints/` and point the notebooks at it with the
`MEDSAM2_CHECKPOINT` environment variable (full path). The notebooks otherwise
default to `MedSAM2_MRI_LiverLesion.pt` above.

Checkpoints are plain `{'model': state_dict}` files, so a fine-tuned checkpoint
from the 3DMPUS-segmentation repo is a drop-in replacement.

## Setup

Install the GPU dependencies (includes the `sam2` package):

```bash
pip install -r requirements.txt -r requirements_GPU.txt
```

Training and evaluation live in the separate **3DMPUS-segmentation** repository.
