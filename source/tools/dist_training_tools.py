import os
import torch
import torch.distributed as dist

from models.instance_segmentation_models import Model
from pipeline.trainer import Trainer


class DistributedWrapper(Model):
    def __init__(self, shared_model):
        super(DistributedWrapper, self).__init__()
        self.shared_model = shared_model

    def forward(self, images: torch.tensor, *targets):
        return self.shared_model(images, *targets)

    def compute_actions(self, images: torch.tensor, num_pokes: int, episode: int, episodes: int):
        return self.shared_model.module.compute_actions(images, num_pokes, episode, episodes)

    def compute_masks(self, images: torch.tensor, threshold: float):
        return self.shared_model.module.compute_masks(images, threshold)


class DummySharedWrapper(torch.nn.Module):
    def __init__(self, model):
        super(DummySharedWrapper, self).__init__()
        self.module = model

    def forward(self, x, *y):
        return self.module(x, *y)


def do_setup_and_start_training(modules, configs, rank, size, device_list, single=False):
    if not single:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29509'
        dist.init_process_group('nccl', rank=rank, world_size=size)

    with torch.cuda.device(device_list[rank]):
        print('initializing model on rank %d'%rank)
        if single:
            shared_model = DummySharedWrapper(modules['model'](configs['model'])).cuda()
        else:
            shared_model = torch.nn.parallel.DistributedDataParallel(modules['model'](configs['model']).cuda(),
                                                                     device_ids=[device_list[rank]],
                                                                     find_unused_parameters=True)
        model = DistributedWrapper(shared_model)
        memory = modules['memory'](configs['memory'])
        loss_function = modules['loss'](configs['loss'])
        trainer_config = configs['trainer']
        trainer_config.log_path = trainer_config.log_path + '%d.log'%rank
        if rank > 0:
            trainer_config.save_frequency = 0
        trainer = Trainer(model, memory, loss_function, trainer_config)
        print('starting training process on rank %d'%rank)
        stats = trainer.train()

    print('done %d' % rank)
    return stats

