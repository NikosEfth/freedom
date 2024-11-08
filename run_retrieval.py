import pickle
import numpy as np
import torch
import time
from utils import *
from utils_features import *
from utils_retrieval import *
import os
import argparse
import open_clip
from pdb import set_trace as st


def parse_args():
    parser = argparse.ArgumentParser(description="Retrieval parameters")
    parser.add_argument(
        "--gpu", default=0, type=int, metavar="gpu", help="Choose a GPU id"
    )
    parser.add_argument(
        "--method",
        choices=["image", "text", "sum", "product", "freedom"],
        type=str,
        default="freedom",
        help="Method",
    )
    parser.add_argument(
        "--dataset",
        choices=["imagenet_r", "nico", "minidn", "ltll"],
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
    parser.add_argument("--source", nargs="+", type=str, help="define source domains")
    parser.add_argument("--target", nargs="+", type=str, help="define target domains")
    parser.add_argument("--kappa", type=int, default=20, help="kappa")
    parser.add_argument("--miu", type=int, default=7, help="miu")
    parser.add_argument("--ni", type=int, default=7, help="ni")
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
    args.device = setup_device(gpu_id=args.gpu)
    model.to(args.device)
    model.eval()

    method = args.method.lower()
    if method == "freedom":
        method += f"_miu={args.miu}_ni={args.ni}_kappa={args.kappa}"
    print(f"Dataset: {args.dataset}, Backbone: {args.backbone}, Method: {method}")

    data_info = prepare_dataset(args)

    metrics = {}
    for idx1, source_domain in enumerate(data_info["domains"]):
        if source_domain in data_info["source"]:
            current_indices = [
                idx
                for idx, item in enumerate(data_info["query_dict"]["domains"])
                if item == source_domain
            ]
            current_query_dict = {}
            current_query_dict["feats"] = data_info["query_dict"]["feats"][
                current_indices, :
            ]
            current_query_dict["classes"] = [
                data_info["query_dict"]["classes"][idx] for idx in current_indices
            ]
            for idx2, target_domain in enumerate(data_info["domains"]):
                if (
                    target_domain in data_info["target"]
                    and source_domain != target_domain
                ):
                    start = time.time()
                    text_feature = text_list_to_features(
                        model, tokenizer, [target_domain], args.device
                    ).squeeze(0)
                    text_feature = text_feature.repeat(
                        (len(current_query_dict["classes"]), 1)
                    )
                    real_text = len(current_query_dict["classes"]) * [target_domain]
                    rankings = calculate_rankings(
                        args,
                        model,
                        tokenizer,
                        current_query_dict["feats"],
                        text_feature,
                        real_text,
                        data_info["database_dict"]["feats"],
                    )
                    metrics[source_domain + "-->" + target_domain], _ = metrics_calc(
                        rankings,
                        target_domain,
                        current_query_dict["classes"],
                        data_info["database_dict"]["classes"],
                        data_info["database_dict"]["domains"],
                        data_info["at"],
                    )
                    print(
                        round(time.time() - start, 1),
                        source_domain + "-->" + target_domain,
                        method,
                        metrics[source_domain + "-->" + target_domain],
                    )
        else:
            print(f"{source_domain} is not in the specified source domains")
    for idx, metric_name in enumerate(data_info["metric_names"]):
        metric_save_dir = os.path.join(
            ".", "results", data_info["metric_name_types"][idx]
        )
        os.makedirs(metric_save_dir, exist_ok=True)
        dict_to_csv(
            metrics,
            os.path.join(
                metric_save_dir,
                f"{args.backbone}_{args.dataset}_{method}_{metric_name}_table.csv",
            ),
            metric_name,
        )


if __name__ == "__main__":
    main()
