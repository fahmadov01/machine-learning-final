"""Hyperparameters and paths for the CNN classifier"""

import os


# Dataset location
# MTSD_ROOT can point anywhere the extracted archives live
# defaults to the current directory
DATASET_ROOT = os.environ.get("MTSD_ROOT", ".")

# HuggingFace mirror with precropped sign patches
# use when the full MTSD download is unavailable
HF_DATASET     = "sparshgarg57/mapillary_traffic_signs"
USE_HF_DATASET = False

# Sample filtering for the local dataset
# drops signs that tend to be mislabeled or too hard to learn from
SKIP_OCCLUDED     = True
SKIP_AMBIGUOUS    = True
SKIP_OUT_OF_FRAME = True
SKIP_OTHER_SIGN   = True   # drops the catch all other sign class

# model
HF_MODEL_NAME = "google/efficientnet-b3"
IMG_SIZE      = 224     # EfficientNet-B3 default input size
CROP_PADDING  = 0.15    # extra padding around bbox crops as a fraction of bbox side

# Training
SEED        = 42
EPOCHS      = 30
BATCH_SIZE  = 64        # drop to 32 if an 8 GB GPU hits OOM
NUM_WORKERS = 8
PIN_MEMORY  = True

# Learning rates
# head gets the higher rate since it starts from scratch.
# backbone gets a lower rate so pretrained features dont wash out
LR_HEAD       = 1e-3
LR_BACKBONE   = 1e-4
WEIGHT_DECAY  = 1e-4
WARMUP_EPOCHS = 2
MIN_LR        = 1e-6

# Regularization
DROPOUT         = 0.3
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA     = 0.2   # set to 0 to  disable mixup

# early stopping
PATIENCE = 7

# Paths
CHECKPOINT_DIR  = "./checkpoints"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
LOG_DIR         = "./runs"
CACHE_DIR       = "./cache"   # pickled annotation index cache

# Training time augmentation
AUGMENT_TRAIN = True
RAND_AUG_N    = 2
RAND_AUG_M    = 9

# Inference
TOP_K = 5
