import pickle
import numpy as np
import torch
import csv
import os
from PIL import Image
from torch.utils.data import Dataset


class ImageDomainLabels(Dataset):
    def __init__(self, input_filename, preprocess, root=None):
        with open(input_filename, "r") as f:
            lines = f.readlines()
        filenames = [line.strip() for line in lines]
        self.images = [name.split(" ")[0] for name in filenames]
        self.domains = [name.split(" ")[1] for name in filenames]
        self.labels = [name.split(" ")[2] for name in filenames]
        self.preprocess = preprocess
        self.root = root

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, str(self.images[idx]))
        images = self.preprocess(Image.open(img_path))
        labels = self.labels[idx]
        domains = self.domains[idx]
        return images, labels, domains, img_path


def save_dataset_features(model, dataloader, save_file, device):
    all_image_features, all_image_filenames, all_image_labels, all_image_domains = (
        [],
        [],
        [],
        [],
    )
    with torch.no_grad():
        iterations = 0
        for images, labels, domains, filenames in dataloader:
            iterations += len(filenames)
            print(iterations, end="\r")
            images = images.to(device)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)
            for idx in range(len(filenames)):
                all_image_filenames.append(filenames[idx])
                all_image_labels.append(labels[idx])
                all_image_domains.append(domains[idx])
        all_image_features = torch.cat(all_image_features, dim=0)
        dict_save = {}
        dict_save["feats"] = all_image_features.data.cpu().numpy()
        dict_save["classes"] = all_image_labels
        dict_save["domains"] = all_image_domains
        dict_save["path"] = all_image_filenames

        with open(save_file, "wb") as f:
            pickle.dump(dict_save, f)


def save_corpus_features(model, tokenizer, corpus_path, save_file, device):
    names = []
    with open(corpus_path, newline="") as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            names.append(row[0])

    all_text_features = []
    all_text = []
    all_text_input = []
    all_tokens_per_words = []
    with torch.no_grad():
        for idx, actual_text in enumerate(names):
            print(idx, end="\r")
            text = []
            text_tokens = tokenizer([actual_text], context_length=model.context_length)
            text.append(text_tokens)

            text = torch.cat(text, dim=0)
            text = text.to(device)
            text_feature = model.encode_text(text)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)

            all_text_features.append(text_feature)
            all_text.append(actual_text)
        all_text_features = torch.cat(all_text_features, dim=0)
        dict_save = {}
        dict_save["feats"] = all_text_features.data.cpu().numpy()
        dict_save["prompts"] = all_text
        with open(save_file, "wb") as f:
            pickle.dump(dict_save, f)


def read_dataset_features(pickle_dir, device):
    with open(pickle_dir, "rb") as f:
        data = pickle.load(f)
    data["feats"] = torch.from_numpy(data["feats"].astype("float32")).float().to(device)
    return data


def read_corpus_features(pickle_dir, device):
    with open(pickle_dir, "rb") as data:
        data_dict = pickle.load(data)
        descr = (
            torch.from_numpy(data_dict["feats"].astype("float32")).float().to(device)
        )
        names = np.array(data_dict["prompts"])
    return descr, names
