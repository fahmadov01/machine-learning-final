import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer
)

ANNOTATIONS_PATH = "../annotations"
IMAGE_DIR = "../images"
MODEL_NAME = "google/vit-base-patch16-224"
OUTPUT_DIR = "./output"
BATCH_SIZE = 8
EPOCHS = 5


"""
    This reads the annotation JSON files and builds a dataset list.

    Each sample has:
    - image filename
    - list of labels (multi-label classification)
"""
def build_samples(path):
    samples = []
    for f in os.listdir(path):
        if not f.endswith(".json"):
            continue

        with open(os.path.join(path, f)) as file:
            data = json.load(file)

        objects = data.get("objects", [])
        objects = [o for o in objects if not o["properties"]["ambiguous"]]

        if len(objects) == 0:
            continue

        labels = list(set(o["label"] for o in objects))
        image_file = f.replace(".json", ".jpg")

        samples.append({"image": image_file, "labels": labels})

    return samples

"""
    This builds a mapping from string labels to integer indices.
"""
def build_label_map(samples):
    all_labels = set()
    for s in samples:
        all_labels.update(s["labels"])
    labels = sorted(all_labels)
    return {l: i for i, l in enumerate(labels)}

"""
    This is a custom dataset that converts the image and labels into:
    - pixel tensors (via ViT processor)
    - multi-hot label vectors
"""
class TrafficSignDataset(Dataset):
    def __init__(self, samples, label_map, image_dir, processor):
        self.samples = samples
        self.label_map = label_map
        self.image_dir = image_dir
        self.processor = processor
        self.num_labels = len(label_map)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        img_path = os.path.join(self.image_dir, s["image"])
        image = Image.open(img_path).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        label_vec = torch.zeros(self.num_labels, dtype=torch.float32)
        for l in s["labels"]:
            label_vec[self.label_map[l]] = 1.0

        inputs["labels"] = label_vec
        return inputs

"""
    This computes the evaluation metric during Trainer evaluation
"""
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    labels = labels.astype(int)

    accuracy = (preds == labels).mean()

    return {"accuracy": accuracy}

#This will test 5 random images and output the results to a function
def test_random_images(model, processor, samples, image_dir, label_map, output_file):
    model.eval()
    id2label = {i: l for l, i in label_map.items()}

    #Choose 5 random images from samples
    chosen = random.sample(samples, min(5, len(samples)))

    with open(output_file, "a") as f:
        f.write("\n===== RANDOM TEST RESULTS =====\n")

        for s in chosen:
            #Open image
            img_path = os.path.join(image_dir, s["image"])
            image = Image.open(img_path).convert("RGB")

            #Input into model
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits.cpu().numpy()[0]

            #Calculate and if probabilty greater than 0.5, add it to predicted
            probs = 1 / (1 + np.exp(-logits))
            preds = (probs > 0.5).astype(int)

            pred_labels = set([id2label[i] for i, v in enumerate(preds) if v == 1])
            true_labels = set(s["labels"])

            correct = pred_labels == true_labels
            
            #Write to output file
            f.write(
                f"Image: {s['image']} | "
                f"Ground Truth: {list(true_labels)} | "
                f"Predicted: {list(pred_labels)} | "
                f"Correct: {correct}\n"
            )

def main():
    # This builds the dataset from annotations
    samples = build_samples(ANNOTATIONS_PATH)

    # This splits the training and validation splits (90/10)
    split = int(0.9 * len(samples))
    train_samples = samples[:split]
    val_samples = samples[split:]
    val_samples = samples[split]

    #This is a ViT image preprocessing pipeline
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    
    #This builds the label to index mapping
    label_map = build_label_map(samples)

    #This creates the datasets
    train_dataset = TrafficSignDataset(train_samples, label_map, IMAGE_DIR, processor)
    val_dataset = TrafficSignDataset(val_samples, label_map, IMAGE_DIR, processor)

    # This loads the pretrained ViT model for multi-label classification
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_map),
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True
    )

    #This sets the local mappings to make the more easily interpretable
    model.config.id2label = {i: l for l, i in label_map.items()}
    model.config.label2id = label_map

    #This is the training configuration
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        do_eval=True,
        eval_strategy="no",
        logging_steps=50,
        learning_rate=2e-5,
        remove_unused_columns=False
    )

    # This is the Hugging Face Trainer abstraction
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    #This trains the model
    trainer.train()

    #This is the evalation after the training
    eval_results = trainer.evaluate()
    print(eval_results)

    #Write evaluation results to file
    output_file = os.path.join(OUTPUT_DIR, "test_results.txt")

    with open(output_file, "w") as f:
        f.write("===== EVALUATION RESULTS =====\n")
        for k, v in eval_results.items():
            f.write(f"{k}: {v}\n")

    #Test on random images
    test_random_images(model, processor, val_samples, IMAGE_DIR, label_map, output_file)

if __name__ == "__main__":
    main()