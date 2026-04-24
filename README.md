# 4622-final

# ImageTransformer (ViT)

## Overview
`ImageTransformer.py` is a script for training an image classification transformer model using fully annotated images and their corresponding annotation files from Mapillary. The script uses a pretrained Vision Transformer (ViT) model from Hugging Face (google/vit-base-patch16-224) and fine-tunes it on the custom dataset using the Trainer API

## Dataset Structure
To be able to run the script, please ensure that you have a folder called images that contains the images and a folder called annotations that contains the annotations. The images can be downloaded from [mapillary.com](mapillary.com). Please make sure that you only get the fully annotated images along with the fully annotated annotations. 

## Requirements

You will need a few libraries to be able to run the script. You can install the required dependencies using pip:

```bash
pip install torch torchvision transformers pillow numpy
```

To run the program, you can simply run python3 ImageTransformers.py. Please ensure you are using Python version 3.12.
