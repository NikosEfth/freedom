import os
import argparse
import shutil
import csv


def parse_args():
    parser = argparse.ArgumentParser(description="Setup the dataset")
    parser.add_argument(
        "--dataset",
        choices=["imagenet_r", "nico", "ltll"],
        type=str,
        help="define path",
    )
    return parser.parse_args()


def rename_space_in_file(root_dir, replacement):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if " " in filename:
                new_filename = filename.replace(" ", replacement)
                os.rename(
                    os.path.join(dirpath, filename), os.path.join(dirpath, new_filename)
                )
                print(
                    f"Renamed file: {os.path.join(dirpath, filename)} -> {os.path.join(dirpath, new_filename)}"
                )


def rename_space_in_path(root_dir, replacement):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            if " " in dirname:
                new_dirname = dirname.replace(" ", replacement)
                os.rename(
                    os.path.join(dirpath, dirname), os.path.join(dirpath, new_dirname)
                )
                print(
                    f"Renamed directory: {os.path.join(dirpath, dirname)} -> {os.path.join(dirpath, new_dirname)}"
                )


def copy_images_from_txt(path):
    with open(os.path.join(path, "imgnet_real_query.txt"), "r") as file:
        rows = file.readlines()

    for row in rows:
        row = row.replace("imgnet", "imagenet_r")
        image_target = row.strip().split(" ")[0]
        image_source = os.path.join(
            path, "imagenet_val", os.path.basename(image_target)
        )
        os.makedirs(os.path.dirname(os.path.join("data", image_target)), exist_ok=True)
        if os.path.exists(image_source):
            shutil.copy(image_source, os.path.join("data", image_target))
        else:
            print(f"Image not found: {source_image_target}")


def make_csv(path):
    in_names_dict = {}
    with open(os.path.join(path, "label_names.csv"), "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        for row in csvreader:
            key, value = row
            in_names_dict[key] = value

    imagenet_r_paths = []
    domain_no = []
    with open(os.path.join(path, "imgnet_targets.txt"), "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=" ")
        for row in csvreader:
            imagenet_r_paths.append(row[0].replace("imgnet", "imagenet_r"))
            domain_no.append(int(row[1]))
    class_name_list = [in_names_dict[x.split("/")[-2]] for x in imagenet_r_paths]
    domain_name_list = [
        (
            "cartoon"
            if x < 1000
            else (
                "origami"
                if x < 2000
                else "toy" if x < 3000 else "sculpture" if x < 4000 else "real"
            )
        )
        for x in domain_no
    ]
    data = list(zip(imagenet_r_paths, domain_name_list, class_name_list))
    with open(os.path.join(path, "full_files.csv"), "w", newline="") as file:
        writer = csv.writer(file, delimiter=" ")
        writer.writerows(data)


def main():
    args = parse_args()
    if args.dataset == "nico":
        rename_space_in_path(root_dir=f"./data/{args.dataset}", replacement="_")
    elif args.dataset == "ltll":
        rename_space_in_path(root_dir=f"./data/{args.dataset}", replacement="")
        rename_space_in_file(root_dir=f"./data/{args.dataset}", replacement="_")
    if args.dataset == "imagenet_r":
        copy_images_from_txt(path=f"./data/{args.dataset}")
        make_csv(path=f"./data/{args.dataset}")


if __name__ == "__main__":
    main()
