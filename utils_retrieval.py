import torch
import numpy as np
import os
from utils import *
from utils_features import *


def prepare_dataset(args):
    query_suffix, database_suffix = "query", "database"
    if args.dataset == "imagenet_r":
        query_suffix, database_suffix = "full", "full"
        domain_change = {
            "real": "photo",
            "cartoon": "cartoon",
            "origami": "origami",
            "toy": "toy",
            "sculpture": "sculpture",
        }
        at = [10, 50]
    elif args.dataset == "nico":
        domain_change = {
            "autumn": "autumn",
            "dim": "dimlight",
            "grass": "grass",
            "outdoor": "outdoor",
            "rock": "rock",
            "water": "water",
        }
        at = [50, 100]
    elif args.dataset == "minidn":
        domain_change = {
            "clipart": "clipart",
            "painting": "painting",
            "real": "photo",
            "sketch": "sketch",
        }
        at = [50, 100]
    elif args.dataset == "ltll":
        query_suffix, database_suffix = "full", "full"
        domain_change = {"New": "Today", "Old": "Archive"}
        at = [5, 10]

    metric_names = ["mAP"] + [f"R@{x}" for x in at] + [f"P@{x}" for x in at]
    metric_name_types = ["mAP"] + ["Recall" for _ in at] + ["Precision" for x in at]
    features_dir = os.path.join("features", f"{args.backbone}_features", args.dataset)
    query_dict = read_dataset_features(
        os.path.join(features_dir, f"{query_suffix}_{args.dataset}_features.pkl"),
        args.device,
    )
    database_dict = read_dataset_features(
        os.path.join(features_dir, f"{database_suffix}_{args.dataset}_features.pkl"),
        args.device,
    )
    domains = list(domain_change.values())

    query_dict["domains"] = replace_domain_names(query_dict["domains"], domain_change)
    database_dict["domains"] = replace_domain_names(
        database_dict["domains"], domain_change
    )
    source, target = domains, domains

    if args.source is not None:
        source = args.source
    if args.target is not None:
        target = args.target

    return {
        "query_dict": query_dict,
        "database_dict": database_dict,
        "domains": domains,
        "source": source,
        "target": target,
        "metric_names": metric_names,
        "metric_name_types": metric_name_types,
        "at": at,
    }


def calculate_rankings(
    args,
    model,
    tokenizer,
    image_features,
    text_features,
    real_text,
    database_features,
):
    if args.method == "image":
        sim_img = image_features @ database_features.t()
        ranks = torch.argsort(sim_img, descending=True)
    elif args.method == "text":
        sim_text = text_features @ database_features.t()
        ranks = torch.argsort(sim_text, descending=True)
    elif args.method == "sum":
        sim_img = image_features @ database_features.t()
        sim_text = text_features @ database_features.t()
        ranks = torch.argsort(sim_img + sim_text, descending=True)
    elif args.method == "product":
        sim_img = image_features @ database_features.t()
        sim_text = text_features @ database_features.t()
        ranks = torch.argsort(torch.mul(sim_img, sim_text), descending=True)
    elif args.method == "freedom":
        corpus_file = os.path.join(
            "features",
            f"{args.backbone}_features",
            "corpus",
            "open_image_v7_class_names.pkl",
        )
        text_corpus_features, real_corpus_text = read_corpus_features(
            corpus_file, args.device
        )
        text_corpus_features = text_corpus_features.detach()
        dim = image_features.shape[1]

        with torch.no_grad():
            sim_img = image_features @ database_features.t()
            _, ranks_img = torch.topk(
                sim_img, args.kappa, dim=1, largest=True, sorted=True
            )

            add_self_feature = False
            image_and_neighbor_features = database_features[ranks_img]
            if torch.min(torch.max(sim_img, 1)[0]).item() < 0.98:
                add_self_feature = True
                print("Adding self feature")
                image_and_neighbor_features = torch.cat(
                    (
                        image_features.unsqueeze(1),
                        image_and_neighbor_features[:, 0:-1, :],
                    ),
                    1,
                )

            top_indices = get_word_indices(
                image_and_neighbor_features, text_corpus_features, 1000, args.ni
            )

            top_indices = top_indices.detach().cpu().numpy()
            indexes_np = np.transpose(top_indices, (1, 0, 2))
            text_list_from_img = real_corpus_text[indexes_np]
            text_list_from_img = text_list_from_img.tolist()

            text_list_from_img = [
                [text_list_from_img[j][i] for j in range(len(text_list_from_img))]
                for i in range(len(text_list_from_img[0]))
            ]
            text_list_from_img, weights = keep_k_most_frequent(
                text_list_from_img, args.miu
            )
            text_list_from_text = [[x] for x in real_text]
            multi_texts = text_to_multi(text_list_from_text, text_list_from_img)
            current_miu = len(weights[0])
            weights = torch.tensor(weights).to(args.device).view(-1, 1)

            multi_texts = invert_levels(multi_texts)
            multi_texts = [item for sublist in multi_texts for item in sublist]
            all_features = text_list_to_features(
                model, tokenizer, multi_texts, args.device, 64
            )

            fused_queries = all_features * weights
            fused_queries = fused_queries.reshape(-1, current_miu, dim)
            fused_queries = fused_queries.sum(dim=1)
            sim_total = fused_queries @ database_features.t()
            ranks = torch.argsort(sim_total, descending=True)
    return ranks.detach().cpu()


def metrics_calc(
    rankings,
    target_domain,
    current_query_classes,
    database_classes,
    database_domains,
    at,
):

    metrics = {}
    class_id_map = {class_name: idx for idx, class_name in enumerate(database_classes)}
    domain_id_map = {
        domain_name: idx for idx, domain_name in enumerate(database_domains)
    }

    database_classes_ids = [class_id_map[class_name] for class_name in database_classes]
    database_domains_ids = [
        domain_id_map[domain_name] for domain_name in database_domains
    ]
    query_classes_ids = [
        class_id_map[class_name] for class_name in current_query_classes
    ]
    target_domain_id = domain_id_map[target_domain]

    database_classes_tensor = torch.tensor(database_classes_ids).to(rankings.device)
    database_domains_tensor = torch.tensor(database_domains_ids).to(rankings.device)
    query_classes_tensor = torch.tensor(query_classes_ids).to(rankings.device)
    target_domain_tensor = torch.tensor(target_domain_id).to(rankings.device)

    class_tensor = (
        database_classes_tensor[rankings]
        == torch.unsqueeze(query_classes_tensor, 1).expand_as(rankings)
    ).float()
    domain_tensor = (database_domains_tensor[rankings] == target_domain_tensor).float()

    correct = domain_tensor * class_tensor
    metrics[f"mAP"], AP_list = compute_map(correct.cpu().numpy())
    for k in at:
        correct_k = correct[:, :k]
        num_correct = torch.sum(correct_k, dim=1)
        num_predicted = torch.sum(torch.ones_like(correct_k), dim=1)
        num_total = torch.sum(correct, dim=1)
        recall = torch.mean(num_correct / (num_total + 1e-5))
        precision = torch.mean(
            num_correct / (torch.minimum(num_total, num_predicted) + 1e-5)
        )
        metrics[f"R@{k}"] = round(recall.item() * 100, 2)
        metrics[f"P@{k}"] = round(precision.item() * 100, 2)
    return metrics, AP_list
