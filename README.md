# 4622-final

Two traffic-sign classifiers fine-tuned on the Mapillary Traffic Sign Dataset (MTSD):

- **`CNN/`** — EfficientNet-B3 convolutional baseline
- **`ImageTransformer.py`** — Vision Transformer (ViT) comparison model

Both train on cropped sign patches from MTSD v2 (400 classes). The CNN directory also contains the saved training/evaluation outputs, the trained checkpoint, and a ready-to-run inference script.

---

# CNN (EfficientNet-B3)

## Overview
`CNN/` fine-tunes `google/efficientnet-b3` (HuggingFace, ImageNet-pretrained) on MTSD sign patches. Two-stage fine-tuning: frozen backbone for 5 epochs, then end to end with a lower backbone LR. Training uses label smoothing, Mixup, and RandAugment. Full hyperparameters in `CNN/config.py`.

## Results

Best checkpoint at epoch 19 of 26.

| Metric | Value |
|---|---|
| Top-1 accuracy (val) | **92.92%** |
| Top-5 accuracy (val) | 99.41% |
| Weighted F1 | 92.72% |
| Macro F1 (386 nonempty classes) | 89.02% |

Evaluation was done on the MTSD val split; the official test split ships without public annotations. Per class metrics in `CNN/eval_results/report_val.csv`, TensorBoard logs in `CNN/runs/`.

## Running inference

```bash
cd CNN
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

PyTorch (pick the build that matches your hardware):
```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CPU-only
pip install torch torchvision
```

The trained checkpoint should be located at `CNN/checkpoints/best_model.pt` It was over GitHub's 100MB size limit and is not in the repo. Please use this Google Drive link to download: https://drive.google.com/file/d/1rWEAtYrTrrwYHj6HWmLVApJspr8NXkIz/view?usp=sharing

Then:
```bash
# single image
python infer.py --checkpoint checkpoints/best_model.pt --image path/to/sign.jpg

# folder of images
python infer.py --checkpoint checkpoints/best_model.pt --folder path/to/signs/

# top-k predictions
python infer.py --checkpoint checkpoints/best_model.pt --image sign.jpg --top-k 10
```

Five precropped test images are committed at `CNN/test_crops/`. Running `python infer.py --checkpoint checkpoints/best_model.pt --folder test_crops/` gets 4/5 correct which is consistent with the 92.9% val accuracy

### Input expectations
The model was trained on tight crops around a single sign with 15% bbox padding. Pass in an image already cropped to one sign for best results.

## Re-evaluating

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt --split val
```
Outputs top-1/top-5 accuracy plus macro precision/recall/F1 to stdout, and writes `eval_results/report_val.csv`. Evaluation on `--split test` will not work (no public annotations).

## Retraining

Retraining requires the full MTSD v2 fully-annotated download (images + annotations, ~47 GB). Extract the archives into `CNN/` (layout below) or point at a custom path via `MTSD_ROOT` or `--mtsd-root`.

```
CNN/
  mtsd_fully_annotated_annotation/mtsd_v2_fully_annotated/
    splits/{train,val,test}.txt
    annotations/<key>.json
  mtsd_fully_annotated_images.train.{0,1,2}/images/<key>.jpg
  mtsd_fully_annotated_images.val/images/<key>.jpg
  mtsd_fully_annotated_images.test/images/<key>.jpg
```

```bash
python train.py                            # local MTSD
python train.py --use-hf                   # HuggingFace mirror
tensorboard --logdir runs/                 # monitor
```

---

# ImageTransformer (ViT)

## Overview
`ImageTransformer.py` is a script for training an image classification transformer model using fully annotated images and their corresponding annotation files from Mapillary. The script uses a pretrained Vision Transformer (ViT) model from Hugging Face (google/vit-base-patch16-224) and fine-tunes it on the custom dataset using the Trainer API

## Dataset Structure
To be able to run the script, please ensure that you have a folder called images that contains the images and a folder called annotations that contains the annotations. It is important that these folders are not in the same directory as the script and are in the parent directory. If you do have them in the same directory, you will have to make some minor adjustments to the script so that it can still run. The images and annotations can be downloaded from [mapillary.com](mapillary.com). Please make sure that you only get the fully annotated images along with the fully annotated annotations.

## Requirements

You will need a few libraries to be able to run the script. You can install the required dependencies using pip:

```bash
pip install torch torchvision transformers pillow numpy
cd ViT
python3 ImageTransformer.py
```

To run the program, you can simply run python3 ImageTransformers.py. Please ensure you are using Python version 3.12.

## Output
The program will write to a folder called output the checkpoints of the model as well as a file called test_results.txt. test_results.txt is a basic text file that will have the evaluation metrics as well as the outputs for the 5 random test images.