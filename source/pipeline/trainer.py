import torch
import numpy as np
from time import time
import os
import logging

from config import TrainerConfig, TestingConfig, global_config
from models.model import Model
from replay_memory.replay_memory import Memory
from pipeline.actor import ActorPool
from losses.losses import LossFunction
from pipeline.tester import Tester


class Trainer:
    """
    The main training pipeline.
    """
    def __init__(self, model: Model, memory: Memory, loss_function: LossFunction,
                 trainer_config: TrainerConfig):
        if os.path.exists(trainer_config.log_path):
            os.remove(trainer_config.log_path)
        logging.basicConfig(filename=trainer_config.log_path, level=logging.DEBUG,
                            format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
        self.model_device = torch.cuda.current_device() if global_config.distributed else global_config.model_gpu

        self.config = trainer_config

        self.model = model

        self.memory = memory

        self.loss_function = loss_function

        self.optimizer = self.init_optimizer()

        self.actors = ActorPool(self.model, self.loss_function, self.memory,
                                trainer_config.num_actors, trainer_config.ground_truth)

        self.model.eval()

    def train(self):

        if self.config.eval_during_train:
            whichval = global_config.val_scenes
            global_config.val_scenes = 1
            tester_val = Tester(self.model, self.memory, self.loss_function, TestingConfig())
            tester_val_path = './tmpvaldata%d' % np.random.randint(0, 10 ** 10, 1)
            tester_val.make_and_save_dataset(500, tester_val_path)
            global_config.val_scenes = 0
            tester_train = Tester(self.model, self.memory, self.loss_function, TestingConfig())
            tester_train_path = './tmptraindata%d' % np.random.randint(0, 10 ** 10, 1)
            tester_train.make_and_save_dataset(500, tester_train_path)
            global_config.val_scenes = whichval
        else:
            tester_val, tester_val_path = None, None
            tester_train, tester_train_path = None, None

        self.actors.start()
        evals = []
        running_val_stats = [0] * (self.loss_function.loss_summary_length + 1)

        t_train = time()

        if self.config.prefill_memory > 0:
            print('prefilling memory')
            num_pokes = self.config.poking_schedule(1, self.config.episodes)
            self.add_batch(self.config.prefill_memory, num_pokes, 1, self.config.episodes)

        self.memory.initialize_loader(self.config.batch_size)

        print('starting training')

        for episode in range(1, self.config.episodes + 1):
            if episode == self.config.unfreeze:
                self.model.toggle_detection_net(True)
            if episode % 50 == 1 and self.config.eval_during_train:
                tester_val.evaluate_coco_metrics(tester_val_path, -2, 'bbox', iou=.5)
                tester_train.evaluate_coco_metrics(tester_train_path, -2, 'bbox', iou=.5)

            if self.config.save_frequency > 0 and episode % self.config.save_frequency == 0:
                torch.save(self.model.state_dict(),
                           self.config.checkpoint_path + '%d.pth' % episode)

            current_pokes = self.config.poking_schedule(episode, self.config.episodes)
            current_num_updates = self.config.update_schedule(episode, self.config.episodes)

            self.optimizer.param_groups[0]['lr'] = self.config.lr_schedule(episode, self.config.episodes)

            new_val_stats = self.add_batch(self.config.new_datapoints_per_episode, current_pokes,
                                           episode, self.config.episodes)

            running_val_stats = [0.7 * old + 0.3 * new for old, new in zip(running_val_stats, new_val_stats)]
            stats = running_val_stats[:]

            stats += self.instance_segmentation_update(current_num_updates)

            print(episode, current_pokes, ', %.4f' * len(stats) % tuple(stats))
            logging.info('%d' % episode + ', %d' % current_pokes + ', %.3f' * len(stats) % tuple(stats))
            evals.append(stats)

        self.actors.stop()
        print(time() - t_train)

        if self.config.eval_during_train:
            os.remove(tester_val_path + '.json')
            os.remove(tester_val_path + '.pickle')
            os.remove(tester_train_path + '.json')
            os.remove(tester_train_path + '.pickle')

        return evals

    def instance_segmentation_update(self, num_updates):
        stats = []
        batch_iterator = self.memory.iterator()
        for _ in range(num_updates):
            batch = next(batch_iterator)

            images = batch['images'].cuda(self.model_device)
            targets = tuple(t.cuda(self.model_device) for t in batch['targets'])

            superpixels = batch['superpixels'] if global_config.superpixels else None

            indices = batch['indices'] if self.memory.prioritized_replay else None

            weights = batch['weights'].cuda(self.model_device)

            self.model.train()
            model_preds = self.model(images, targets)

            losses = self.loss_function(model_preds, targets, weights, superpixels)

            if self.loss_function.prioritized_replay:
                losses, priorities = losses
            else:
                priorities = None

            if priorities is not None and indices is not None:
                self.memory.update_priorities(indices, priorities)

            stats.append(np.array([loss.item() for loss in losses]))
            loss = sum(losses)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.eval()

        stats = list(np.stack(stats).mean(axis=0))
        return stats

    def add_batch(self, batch_size, num_pokes, episode, episodes):
        batch, stats = self.actors.make_batch(batch_size, num_pokes, episode, episodes)
        self.memory.add_batch(batch)
        return stats

    def init_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=self.config.weight_decay)
