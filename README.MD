# Dual-Branch US Tumour Classification Framework









## Environment Setup

```bash
# Create and activate a conda environment (Python 3.10+)
conda create -n us_clf python=3.10 -y
conda activate us_clf

# Install PyTorch (choose the build matching your CUDA version)
# Example for CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

---

## Dataset Preparation

Place three `.npy` files in `data/`:

| File            | Shape            | Description                          |
|-----------------|------------------|--------------------------------------|
| `images.npy`    | `(N, H, W)` *    | Grayscale US images                  |
| `masks.npy`     | `(N, H, W)` *    | Binary tumour segmentation masks     |
| `labels.npy`    | `(N,)`           | Integer labels: 0 = benign, 1 = malignant |

\* Shapes `(N, H, W, 1)` and `(N, 1, H, W)` are also accepted.  
Images are automatically resized to `224×224` and normalised to `[0, 1]`.

---

## Training

```bash
python train.py \
    --images_path data/images.npy \
    --masks_path  data/masks.npy  \
    --labels_path data/labels.npy \
    --augment
```


## Evaluation Only

To evaluate a saved model without re-training, load the checkpoint manually:

```python
import torch, argparse
from models.fusion_model import build_model
from trainers.trainer import Trainer
from datasets.ultrasound_dataset import load_npy_data, build_dataloaders, create_single_split

# Rebuild args from saved config.json
import json
args = argparse.Namespace(**json.load(open("outputs/exp/config.json")))

images, masks, labels = load_npy_data(args.images_path, args.masks_path, args.labels_path)
split = create_single_split(labels, seed=args.seed)
loaders = build_dataloaders(images, masks, labels, split, batch_size=args.batch_size, augment=False)

model = build_model(args)
trainer = Trainer(model, loaders["train"], loaders["val"], loaders["test"], args, "eval_run")
y_true, y_pred, y_prob = trainer.test()
```

---


### Reproducibility

- All experiments set Python/NumPy/PyTorch seeds via `utils.reproducibility.set_seed`.
- Split indices are saved to `splits.json` for exact recreation.
- `torch.backends.cudnn.deterministic = True` is enforced (some GPU ops may remain non-deterministic at the kernel level).




