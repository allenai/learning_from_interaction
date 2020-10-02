import os
import json
from datetime import datetime

from pycocotools.mask import area, toBbox

from tools.logger import LOGGER


def save_coco_dataset(dataset_file, output_folder, classes=("light", "medium", "heavy"), force=False):
    def get_dicts(jsonfile):
        with open(jsonfile, "r") as f:
            res = json.load(f)
        return res

    def data_to_coco(data, classes):
        res = dict(
            info=dict(
                date_created=datetime.now().strftime("%Y%m%d%H%M%S"),
                description="Automatically generated COCO json file",
            ),
            categories=[dict(id=it, name=cl) for it, cl in enumerate(classes)],
            images=[],
            annotations=[],
        )

        for ep in data:
            res["images"].append(dict(
                id=ep["image_id"],
                width=ep["width"],
                height=ep["height"],
                file_name=""
            ))

            for ann in ep["annotations"]:
                seg = ann["segmentation"]
                res["annotations"].append(dict(
                    id=len(res["annotations"]) + 1,
                    image_id=ep["image_id"],
                    bbox=list(toBbox(seg)),
                    area=float(area(seg)),
                    iscrowd=0,
                    category_id=ann["category_id"],
                    segmentation=seg,
                ))

        return res

    dataset_base = os.path.basename(dataset_file)
    json_file_name = os.path.join(output_folder, dataset_base.replace(".json", "__coco_format.json"))
    if os.path.exists(json_file_name) and not force:
        LOGGER.info("skipping conversion; {} already exists".format(json_file_name))
        return json_file_name

    json_dict = data_to_coco(get_dicts(dataset_file), classes)
    with open(json_file_name, "w") as f:
        json.dump(json_dict, f)
    LOGGER.info("COCO gt annotations saved to {}".format(json_file_name))

    return json_file_name
