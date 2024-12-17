import numpy as np
import torch
import sys
import pandas as pd


def setup_device(gpu_id):
    if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU: {gpu_id}")
    else:
        print(f"GPU {gpu_id} not available. Using CPU instead.")
        device = torch.device("cpu")
    return device


def dict_to_csv(data, directory, metric):
    metric_values = {}
    for key, all_metrics in data.items():
        source, target = key.split("-->")
        if source not in metric_values:
            metric_values[source] = {}
        metric_values[source][target] = all_metrics[metric]
    metric_df = pd.DataFrame.from_dict(metric_values, orient="index")
    metric_df = metric_df.sort_index().sort_index(axis=1)
    metric_df["average"] = metric_df.mean(axis=1)
    metric_df.loc["average"] = metric_df.mean(axis=0)
    metric_df = metric_df.round(2)
    metric_df.to_csv(directory)


def text_list_to_features(model, tokenizer, text_list, device, batch_size=1):
    text_features = []
    text_tokens = []
    num_batches = len(text_list) // batch_size
    with torch.no_grad():
        for i in range(num_batches + 1):
            if i == num_batches:
                batch = text_list[i * batch_size :]
            else:
                batch = text_list[i * batch_size : (i + 1) * batch_size]
            if len(batch) > 0:
                text_tokens = tokenizer(batch, context_length=model.context_length).to(
                    device
                )
                text_feature = model.encode_text(text_tokens)
                text_feature = text_feature / text_feature.norm(dim=1, keepdim=True)
                text_feature = text_feature.detach().to(torch.float32)
                text_features.append(text_feature)

    return torch.cat(text_features, dim=0)


def get_word_indices(images, text_corpus_features, batch, ni):

    num_samples = images.size(0)
    all_top_indices = []

    for i in range(0, num_samples, batch):
        batch_features = images[i : i + batch]
        sim = torch.matmul(batch_features, text_corpus_features.t())
        _, top_indices = torch.topk(sim, ni, dim=2, largest=True, sorted=True)
        all_top_indices.append(top_indices)
    all_top_indices = torch.cat(all_top_indices, dim=0)

    return all_top_indices


def replace_domain_names(input_list, mapping_dict):
    updated_list = []
    for element in input_list:
        if element in mapping_dict:
            updated_list.append(mapping_dict[element])
        else:
            updated_list.append(element)
    return updated_list


def keep_k_most_frequent(nested_list_of_text, miu):
    labels = []
    weights = []
    frequencies = []
    for idx1, query in enumerate(nested_list_of_text):
        text_counts = {}
        for idx2, query_neighbors in enumerate(query):
            for idx3, text in enumerate(query_neighbors):
                if text in text_counts:
                    text_counts[text] += 1
                else:
                    text_counts[text] = 1
                if text == "":
                    text_counts[text] = 0
        most_common_texts = sorted(
            text_counts.keys(), key=lambda x: text_counts[x], reverse=True
        )[:miu]
        most_common_texts_values = [text_counts[key] for key in most_common_texts]
        frequencies.append(most_common_texts_values)
        most_common_texts_values = [
            x / max(max(most_common_texts_values), 0.0001)
            for x in most_common_texts_values
        ]
        labels.append(most_common_texts)
        weights.append(most_common_texts_values)

    max_len_text = max(len(x) for x in labels)
    max_len_weights = max(len(x) for x in weights)
    labels = [x + [""] * (max_len_text - len(x)) for x in labels]
    weights = [x + [0] * (max_len_weights - len(x)) for x in weights]
    return labels, weights


def text_to_multi(text_list_from_text, text_list_from_img):
    text = [[] for _ in range(len(text_list_from_img))]
    for idx in range(len(text_list_from_text)):
        for domain in text_list_from_text[idx]:
            for clas in text_list_from_img[idx]:
                text[idx].append(clas + " " + domain)
    return invert_levels(text)


def invert_levels(input_list):
    if not input_list:
        return []
    d1 = len(input_list)
    d2 = len(input_list[0])

    if any(len(sublist) != d2 for sublist in input_list):
        raise ValueError("Sublists do not have consistent lengths")
    transposed = [[input_list[i][j] for i in range(d1)] for j in range(d2)]

    return transposed

# From https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/1e66a417afa2247edde6d35f3a9a2a465778a3a8/cirtorch/utils/evaluate.py#L3
def compute_ap(ranks, nres):
    nimgranks = len(ranks)
    ap = 0
    recall_step = 1.0 / (nres + 1e-5)
    for j in np.arange(nimgranks):
        rank = ranks[j]
        if rank == 0:
            precision_0 = 1.0
        else:
            precision_0 = float(j) / rank
        precision_1 = float(j + 1) / (rank + 1)
        ap += (precision_0 + precision_1) * recall_step / 2.0
    return ap


def compute_map(correct):
    map = 0.0
    nq = correct.shape[0]

    ap_list = []

    for i in np.arange(nq):
        pos = np.where(correct[i] != 0)[0]
        ap = compute_ap(pos, len(pos))
        ap_list.append(ap)
        map = map + ap
    map = map / (nq)
    return np.around(map * 100, decimals=2), ap_list
