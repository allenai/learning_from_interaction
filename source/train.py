import os
import argparse

from models.clustering_models import ClusteringModel
from pipeline.trainer import Trainer
from pipeline.tester import Tester
from replay_memory.replay_pil import ReplayPILDataset
from losses.clustering_losses import MaskAndMassLoss
from config import MemoryConfigPIL, MaskAndMassLossConfig, TrainerConfig, TestingConfig, ClusteringModelConfig
from config import global_config, actor_config


def get_args():
    parser = argparse.ArgumentParser(
        description="self-supervised-objects training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="required output model folder name",
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
        "-g",
        "--model_gpu",
        required=False,
        default=0,
        type=int,
        help="gpu id to run model",
    )
    parser.add_argument(
        "-a",
        "--actors_gpu",
        required=False,
        default=1,
        type=int,
        help="gpu id to run AI2-THOR actors",
    )
    parser.add_argument(
        "-p",
        "--checkpoint_prefix",
        required=False,
        default="clustering_model_weights_",
        type=str,
        help="prefix for checkpoints in output folder",
    )
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    print("Running train with args {}".format(args))

    output_folder = os.path.normpath(args.output_folder)
    dataset_folder = os.path.normpath(args.dataset_folder)
    dataset = args.dataset

    os.makedirs(output_folder, exist_ok=True)

    assert os.path.isdir(output_folder), 'Output folder does not exist'
    assert os.path.isdir(dataset_folder), 'Dataset folder does not exist'
    assert dataset in [0, 1], 'Dataset argument should be either 0 (NovelObjects) or 1 (NovelSpaces)'

    global_config.model_gpu = args.model_gpu
    global_config.actor_gpu = args.actors_gpu

    actor_config.data_files = [['NovelObjects__train.json',
                                'NovelObjects__test.json'],
                               ['NovelSpaces__train.json',
                                'NovelSpaces__test.json']][dataset]
    actor_config.data_files = [os.path.join(dataset_folder, fn) for fn in actor_config.data_files]

    loss_function = MaskAndMassLoss(MaskAndMassLossConfig())
    model = ClusteringModel(ClusteringModelConfig()).cuda(global_config.model_gpu)

    trainer_config = TrainerConfig()
    trainer_config.log_path = os.path.join(output_folder, 'training_log.log')

    trainer = Trainer(model, ReplayPILDataset(MemoryConfigPIL()), loss_function, trainer_config)

    print('Running instance segmentation only pre-training')

    actor_config.instance_only = True
    loss_function.config.instance_only = True
    trainer_config.checkpoint_path = os.path.join(output_folder, args.checkpoint_prefix + 'inst_only_')

    model.toggle_mass_head(False)

    trainer.train()

    print('Training with force prediction')

    actor_config.instance_only = False
    loss_function.config.instance_only = False
    trainer_config.checkpoint_path = os.path.join(output_folder, args.checkpoint_prefix)
    trainer_config.update_schedule = lambda episode, episodes: int(15 + 20 * episode / episodes)
    trainer_config.poking_schedule = lambda episode, episodes: 10
    trainer.memory = ReplayPILDataset(MemoryConfigPIL())

    model.toggle_mass_head(True)
    model.toggle_detection_net(False)
    trainer_config.unfreeze = 100

    trainer.train()

    print('Testing on small subset of Val and preparing some illustrations')

    tester = Tester(model, ReplayPILDataset(MemoryConfigPIL()), loss_function, TestingConfig())

    tester.illustrate_poking_and_predictions(os.path.join(output_folder, 'illustrations/'), 50, -.2)
    tester.evaluate_coco_metrics(300, threshold=-100, annotation_type='bbox', iou=.5)
