import os
import argparse
import zipfile
import json
import numpy as np


def extract_data(data_dir):
    print("processing ShapeNetCore Data")
    _, dir_names, file_names = next(os.walk(data_dir))
    for file_name in file_names:
        with zipfile.ZipFile(os.path.join(data_dir, file_name)) as zf:
            print("extracting file '%s' ... " % file_name, end="", flush=True)
            category_name, _ = os.path.splitext(file_name)
            extracted_dir_name = category_name + '_depth_rgb'
            if extracted_dir_name in dir_names:
                print("already extracted, skip")
            else:
                for file in zf.namelist():
                    if file.startswith(extracted_dir_name):
                        zf.extract(file, data_dir)
                print("done")


def process_data(data_dir, index_dir, train_ratio, test_ratio):
    print("Processing ShapeNetCore Data")
    os.makedirs(index_dir, exist_ok=True)
    all_data = dict()
    labels = []
    _, dir_names, _ = next(os.walk(data_dir))
    for dir_name in dir_names:
        if not dir_name.endswith('_depth_rgb'):
            continue
        labels.append(dir_name)
        category_name, _, _ = dir_name.rsplit('_', 2)
        print("Found category %s, processing ... " % category_name, end="", flush=True)
        models = set()
        view_points = set()
        images = os.listdir(os.path.join(data_dir, dir_name))
        dir_data = dict()
        for image in images:
            image_name, _ = os.path.splitext(image)
            _, model, _, view_point = image_name.rsplit('_', 3)
            models.add(model)
            view_points.add(view_point)
        assert len(models) * len(view_points) == len(images)
        for model in models:
            dir_data[model] = [None] * len(view_points)
        for image in images:
            image_name, _ = os.path.splitext(image)
            _, model, _, view_point = image_name.rsplit('_', 3)
            dir_data[model][int(view_point)] = os.path.join(data_dir, dir_name, image)
        all_data[category_name] = dir_data
        print('done')
    print('Collecting information ... ', end="", flush=True)
    train_index = dict()
    test_index = dict()
    for category_name, dir_data in all_data.items():
        sep = int(train_ratio * len(dir_data))
        index = list(dir_data.values())
        np.random.shuffle(index)
        train_index[category_name] = index[:sep]
        test_index[category_name] = index[sep:]
    print('done')
    print('Writing to disk ... ', end="", flush=True)
    with open(os.path.join(index_dir, 'labels.json'), "w") as file:
        json.dump(labels, file)
    with open(os.path.join(index_dir, 'train_index.json'), "w") as file:
        json.dump(train_index, file)
    with open(os.path.join(index_dir, 'test_index.json'), "w") as file:
        json.dump(test_index, file)
    print('done')


def process_nyud(data_dir, index_dir):
    print("Processing nyud data")
    os.makedirs(index_dir, exist_ok=True)
    _, dir_names, _ = next(os.walk(data_dir))
    index = dict()
    for dir_name in dir_names:
        images = os.listdir(os.path.join(data_dir, dir_name))
        index[dir_name] = [os.path.join(data_dir, dir_name, image) for image in images if
                           os.path.splitext(image)[0].endswith("depth")]
    with open(os.path.join(index_dir, 'nyud_index.json'), "w") as file:
        json.dump(index, file)
    print("done")


def main(args):
    if not args.nyud:
        data_dir = args.data_dir
        index_dir = args.index_dir
        skip_extract = args.skip_extract
        if not skip_extract:
            extract_data(data_dir)
        train_ratio = args.train_ratio
        test_ratio = args.test_ratio
        assert train_ratio + test_ratio == 1.0, \
            "The sum of all ratios is not equal to 1"
        process_data(data_dir, index_dir, train_ratio, test_ratio)
    else:
        nyud_data_dir = args.nyud_data_dir
        index_dir = args.index_dir
        process_nyud(nyud_data_dir, index_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        help="data dir",
        default='/home/fz/Downloads/nonbenchmark',
    )
    parser.add_argument(
        "--index_dir",
        help="index dir",
        default='/home/fz/Downloads/nonbenchmark/index'
    )
    parser.add_argument(
        "--skip_extract",
        help="whether skip extract process or not",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--train_ratio",
        help="the ratio of data used in training",
        type=float,
        default=0.9
    )
    parser.add_argument(
        "--test_ratio",
        help="the ratio of data used in testing",
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--nyud",
        help="whether process nyud data",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--nyud_data_dir",
        help="nyud data dir",
        default='/home/fz/Downloads/nyud'
    )
    args = parser.parse_args()
    main(args)
