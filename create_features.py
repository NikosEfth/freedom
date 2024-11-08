import os
import numpy as np
import torch
import pickle
import csv
import argparse
import open_clip
from PIL import Image
from torch.utils.data import DataLoader
from utils_features import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Frature extraction parameters")
    parser.add_argument(
        "--dataset",
        choices=["corpus", "imagenet_r", "nico", "minidn", "ltll"],
        type=str,
        help="define dataset",
    )
    parser.add_argument(
        "--backbone",
        choices=["clip", "siglip"],
        default="clip",
        type=str,
        help="choose the backbone",
    )
    parser.add_argument("--batch", default=512, type=int, help="choose a batch size")
    parser.add_argument(
        "--gpu", default=0, type=int, metavar="gpu", help="Choose a GPU id"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.backbone == "siglip":
        model, preprocess = open_clip.create_model_from_pretrained(
            "hf-hub:timm/ViT-L-16-SigLIP-256"
        )
        tokenizer = open_clip.get_tokenizer("hf-hub:timm/ViT-L-16-SigLIP-256")
    elif args.backbone == "clip":
        model, preprocess = open_clip.create_model_from_pretrained("ViT-L/14", "openai")
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
    device = setup_device(gpu_id=args.gpu)
    model.to(device)
    model.eval()

    save_dir = os.path.join("features", f"{args.backbone}_features", args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    if args.dataset == "corpus":
        corpus_path = "./data/open_image_v7_class_names.csv"
        save_file = os.path.join(save_dir, "open_image_v7_class_names.pkl")
        save_corpus_features(
            model=model,
            tokenizer=tokenizer,
            corpus_path=corpus_path,
            save_file=save_file,
            device=device,
        )
    else:
        dataset_types = ["query", "database"]
        if args.dataset in ["imagenet_r", "ltll"]:
            dataset_types = ["full"]
        for dataset_type in dataset_types:
            path = os.path.join(".", "data", args.dataset, f"{dataset_type}_files.csv")
            dataset = ImageDomainLabels(path, root="./data", preprocess=preprocess)
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
            )
            save_file = os.path.join(
                save_dir, f"{dataset_type}_{args.dataset}_features.pkl"
            )
            save_dataset_features(
                model=model, dataloader=dataloader, save_file=save_file, device=device
            )


if __name__ == "__main__":
    main()
