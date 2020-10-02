import numpy as np
import torch
from torchvision import transforms
import pickle

from config import MemoryConfigTensor, global_config
from replay_memory.replay_memory import Memory


class ReplayTensorDataset(Memory):
    """
    In this class, all output of the Actor (images and targets) are saved as tensors. No data augmentation happens
    in the memory's iterator. This should be the fastest (most lightweight) way to implement the Memory class.
    """

    def __init__(self, memory_config: MemoryConfigTensor):
        super(ReplayTensorDataset, self).__init__()
        self.config = memory_config
        self.to_tensor = transforms.ToTensor()
        memory = tuple(torch.zeros((memory_config.capacity,) + shape, dtype=dtype) for shape, dtype in
                       zip(memory_config.sizes, memory_config.dtypes))
        self.memory = GrowingTensorDataset(memory, self)
        self.loader = None
        self.last = 0
        self.max = 0
        if memory_config.warm_start_memory is not None:
            self.load_memory(memory_config.warm_start_memory)

    def add_batch(self, batch):
        for datapoint in zip(*batch):
            if self.last == self.config.capacity:
                self.last = 0
            datapoint = self.base_transform(datapoint)
            for tensor, new_entry in zip(self.memory.tensors, datapoint):
                tensor[self.last] = new_entry
            self.last += 1
            self.max = max(self.last, self.max)

    def initialize_loader(self, batch_size):
        self.loader = torch.utils.data.DataLoader(self.memory, shuffle=True, batch_size=batch_size, pin_memory=True,
                                                  num_workers=self.config.num_workers)

    def iterator(self):
        if global_config.superpixels:
            return iter(dict(images=batch[0], targets=batch[1:-1], superpixels=batch[-1]) for batch in self.loader)
        return iter(dict(images=batch[0], targets=batch[1:]) for batch in self.loader)

    def load_memory(self, path):
        tensors = pickle.load(open(path, 'rb'))
        tensors = [tensor[-self.config.capacity:] for tensor in tensors]
        size = tensors[0].shape[0]
        for t1, t2 in zip(self.memory.tensors, tensors):
            t1[:size] = t2
        self.max = size
        self.last = size

    def save_memory(self, path):
        tensors = [tensor[:self.max] for tensor in self.memory.tensors]
        file = open(path, 'wb')
        pickle.dump(tensors, file)
        file.close()

    def base_image_transform(self, image):
        if global_config.depth:
            image = np.concatenate([image[0], image[1][..., None]], axis=2)
        return self.to_tensor(image)

    def base_transform(self, datapoint):
        image = self.base_image_transform(datapoint[0])
        targets_and_superpixels = [x for x in datapoint[1]] + ([datapoint[2]] if global_config.superpixels else [])
        return [image] + [torch.from_numpy(x) for x in targets_and_superpixels]

    def update_priorities(self, batch_indices: list, priorities: list):
        """
        :param batch_indices: The indices of the images in the mini-batch on which the loss was just evaluated
        :param priorities: The losses correspondig to the individual images of the minibatch
        :return: Nothing. Update the priorities of the images, in case prioritizes replay is used.
        """
        raise NotImplementedError


class GrowingTensorDataset(torch.utils.data.TensorDataset):
    """
    Trivial modification of pytorch's inbuilt TensorDataset to allow for dataset size that grows during training
    """

    def __init__(self, memory, owner):
        super(GrowingTensorDataset, self).__init__(*memory)
        self.owner = owner

    def __len__(self):
        return self.owner.max
