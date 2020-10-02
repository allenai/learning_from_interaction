import os
import argparse
import json

import torch

from models.clustering_models import ClusteringModel
from pipeline.evaluator import Evaluator
from replay_memory.replay_pil import ReplayPILDataset
from config import MemoryConfigPIL, TestingConfig, ClusteringModelConfig
from config import global_config
from tools.logger import init_logging, LOGGER
from tools.coco_tools import save_coco_dataset


def get_args():
    parser = argparse.ArgumentParser(
        description="self-supervised-objects eval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model_folder",
        type=str,
        help="required trained model folder name",
    )
    parser.add_argument(
        "dataset_folder",
        type=str,
        help="required dataset folder name",
    )
    parser.add_argument(
        "dataset",
        type=int,
        help="required dataset type should be 0 (NovelObjects) or 1 (NovelSpaces) and match the one used for training",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        required=False,
        default=900,
        type=int,
        help="checkpoint to evaluate",
    )
    parser.add_argument(
        "-g",
        "--model_gpu",
        required=False,
        default=0,
        type=int,
        help="gpu id to run model",
    )
    parser.add_argument(
        "-l",
        "--loaders_gpu",
        required=False,
        default=0,
        type=int,
        help="gpu id to run thor data loaders",
    )
    parser.add_argument(
        "-i",
        "--interaction_threshold",
        required=False,
        default=-100.0,
        type=float,
        help="interaction logits threshold",
    )
    parser.add_argument(
        "-p",
        "--checkpoint_prefix",
        required=False,
        default="clustering_model_weights_",
        type=str,
        help="prefix for checkpoints in output folder",
    )
    parser.add_argument(
        "-d",
        "--det_file",
        required=False,
        default=None,
        type=str,
        help="precomputed detections result",
    )
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    init_logging()
    LOGGER.info("Running eval with args {}".format(args))

    output_folder = os.path.normpath(args.model_folder)
    dataset_folder = os.path.normpath(args.dataset_folder)
    dataset = args.dataset

    assert os.path.isdir(output_folder), 'Output folder does not exist'
    assert os.path.isdir(dataset_folder), 'Dataset folder does not exist'
    assert dataset in [0, 1], 'Dataset argument should be either 0 (NovelObjects) or 1 (NovelSpaces)'

    results_folder = os.path.join(output_folder, "inference")
    os.makedirs(results_folder, exist_ok=True)
    LOGGER.info("Writing output to {}".format(results_folder))

    data_file = ['NovelObjects__test.json', 'NovelSpaces__test.json'][dataset]
    data_path = os.path.join(dataset_folder, data_file)
    coco_gt_path = save_coco_dataset(data_path, results_folder)

    if args.det_file is None:
        global_config.model_gpu = args.model_gpu
        global_config.actor_gpu = args.loaders_gpu
        model = ClusteringModel(ClusteringModelConfig()).cuda(global_config.model_gpu)

        cp_name = os.path.join(output_folder, "{}{}.pth".format(args.checkpoint_prefix, args.checkpoint))
        LOGGER.info("Loading checkpoint {}".format(cp_name))
        model.load_state_dict(torch.load(cp_name, map_location="cpu"))

        id = "{}__cp{}".format(os.path.basename(output_folder), args.checkpoint)

        eval = Evaluator(model, ReplayPILDataset(MemoryConfigPIL()), loss_function=None, tester_config=TestingConfig())
        det_file = eval.inference(data_path, results_folder, args.interaction_threshold, id, interactable_classes=[0, 1, 2])
    else:
        det_file = args.det_file
        LOGGER.info("Using precomputed detections in {}".format(det_file))

    results = {}
    for anno_type in ['bbox', 'segm', 'mass']:
        results.update(Evaluator.evaluate(coco_gt_path, det_file, annotation_type=anno_type))

    results_file = det_file.replace("_inf.json", "_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4, sort_keys=True)
    LOGGER.info("Full results saved in {}".format(results_file))

    LOGGER.info("Eval done")
