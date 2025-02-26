import argparse
import os.path as path
import yaml
from tqdm.auto import trange
from os import listdir
from pathlib import Path
import os
import ujson as json  # better to use "imports ujson as json" for the best performance

import uuid
import logging

from PIL import Image
from typing import Optional, Tuple
from urllib.request import (
    pathname2url,
)  # for converting "+","*", etc. in file paths to appropriate urls

from label_studio_sdk.converter.imports.label_config import generate_label_config

logger = logging.getLogger("root")
default_image_root_url = "/data/local-files/?d=images"


def convert_yolo_to_ls(
    input_dir,
    out_file,
    to_name="image",
    from_name="label",
    out_type="annotations",
    image_root_url=default_image_root_url,
    image_ext=".jpg,.jpeg,.png",
    image_dims: Optional[Tuple[int, int]] = None,
):
    """Convert YOLO labeling to Label Studio JSON
    :param input_dir: directory with YOLO where images, labels, notes.json are located
    :param out_file: output file with Label Studio JSON tasks
    :param to_name: object name from Label Studio labeling config
    :param from_name: control tag name from Label Studio labeling config
    :param out_type: annotation type - "annotations" or "predictions"
    :param image_root_url: root URL path where images will be hosted, e.g.: http://example.com/images
    :param image_ext: image extension/s - single string or comma separated list to search, eg. .jpeg or .jpg, .png and so on.
    :param image_dims: image dimensions - optional tuple of integers specifying the image width and height of *all* images in the dataset. Defaults to opening the image to determine it's width and height, which is slower. This should only be used in the special case where you dataset has uniform image dimesions.
    """

    logger.info("Reading YOLO notes and categories from %s", input_dir)

    # build categories=>labels dict
    notes_file = os.path.join(input_dir, "classes.txt")
    with open(notes_file) as f:
        lines = [line.strip() for line in f.readlines()]
    categories = {i: line for i, line in enumerate(lines)}
    logger.info(f"Found {len(categories)} categories")

    # generate and save labeling config
    label_config_file = out_file.replace(".json", "") + ".label_config.xml"
    generate_label_config(
        categories,
        {from_name: "PolygonLabels"},
        to_name,
        from_name,
        label_config_file,
    )

    # define directories
    labels_dir = os.path.join(input_dir, "labels")
    images_dir = os.path.join(input_dir, "images")
    logger.info("Converting labels from %s", labels_dir)

    # build array out of provided comma separated image_extns (str -> array)
    image_ext = [x.strip() for x in image_ext.split(",")]
    logger.info(f"image extensions->, {image_ext}")

    # loop through images
    for f in os.listdir(images_dir):
        image_file_found_flag = False
        for ext in image_ext:
            if f.endswith(ext):
                image_file = f
                image_file_base = os.path.splitext(f)[0]
                image_file_found_flag = True
                break
        if not image_file_found_flag:
            continue

        image_root_url += "" if image_root_url.endswith("/") else "/"
        task = {
            "data": {
                # eg. '../../foo+you.py' -> '../../foo%2Byou.py'
                "image": image_root_url
                + str(pathname2url(image_file))
            }
        }

        # define coresponding label file and check existence
        label_file = os.path.join(labels_dir, image_file_base + ".txt")

        if os.path.exists(label_file):
            task[out_type] = [
                {
                    "result": [],
                    "ground_truth": False,
                }
            ]

            # read image sizes
            if image_dims is None:
                # default to opening file if we aren't given image dims. slow!
                with Image.open(os.path.join(images_dir, image_file)) as im:
                    image_width, image_height = im.size
            else:
                image_width, image_height = image_dims

            with open(label_file) as file:
                # convert all bounding boxes to Label Studio Results
                lines = file.readlines()
                for line in lines:
                    values = line.split()
                    label_id = values[0]
                    coords = [float(x)*100 for x in values[1:]]
                    grouped_coords = [coords[i: i + 2]
                                      for i in range(0, len(coords), 2)]
                    score = None
                    item = {
                        "id": uuid.uuid4().hex[0:10],
                        "type": "polygonlabels",
                        "value": {
                            "points": grouped_coords,
                            "polygonlabels": [categories[int(label_id)]],
                        },
                        "to_name": to_name,
                        "from_name": from_name,
                        "image_rotation": 0,
                        "original_width": image_width,
                        "original_height": image_height,
                    }
                    if score:
                        item["score"] = score
                    task[out_type][0]["result"].append(item)

                with open(os.path.join(labels_dir, image_file_base + ".json"), "w") as f:
                    json.dump(task, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-s3", "--s3-url")
    parser.add_argument("-is", "--is-split", action="store_true")
    args = parser.parse_args()
    input_folder = args.input

    if args.is_split:
        search_folders = ["train", "valid", "test"]
        subfolders = ["images", "labels"]
        tmp_path = path.join(args.input, "tmp")
        for sub in subfolders:
            Path(path.join(tmp_path, sub)).mkdir(exist_ok=True, parents=True)

        config = yaml.safe_load(open(path.join(args.input, "data.yaml"), "r"))
        classes = config["names"]
        with open(path.join(tmp_path, "classes.txt"), "w") as f:
            f.write("\n".join(classes))

        for folderidx in trange(len(search_folders), desc="Folders"):
            folder = search_folders[folderidx]
            for subidx in trange(len(subfolders), desc="Subfolders", leave=False):
                sub = subfolders[subidx]
                files = listdir(path.join(args.input, folder, sub))
                for fileidx in trange(len(files), desc="Files", leave=False):
                    filename = files[fileidx]
                    filepath = path.join(args.input, folder, sub, filename)
                    content = None
                    with open(filepath, "rb") as f:
                        content = f.read()
                    if content is None:
                        continue
                    with open(path.join(tmp_path, sub, filename), "wb") as f:
                        f.write(content)
        input_folder = tmp_path

    convert_yolo_to_ls(input_folder, args.output,
                       image_root_url=args.s3_url)


if __name__ == "__main__":
    main()
