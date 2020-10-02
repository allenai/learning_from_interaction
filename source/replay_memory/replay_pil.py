import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as tf
import pickle

from config import MemoryConfigPIL, global_config
from replay_memory.replay_memory import Memory


class ReplayPILDataset(Memory):
    """
    A DIY memory class where images are saved in PIL format, allowing for flipping and color jittering
    as data augmentation during training. Actual memory use is not very efficient/fast.
    """

    def __init__(self, memory_config: MemoryConfigPIL):
        super(ReplayPILDataset, self).__init__()
        self.prioritized_replay = memory_config.prioritized_replay
        self.config = memory_config
        self.images = [None] * memory_config.capacity
        self.depths = [None] * memory_config.capacity if global_config.depth else None
        self.targets = [None] * memory_config.capacity
        self.superpixels = [None] * memory_config.capacity if global_config.superpixels else None
        self.priorities = np.zeros(memory_config.capacity) if self.prioritized_replay else None
        self.last = 0
        self.max = 0
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        self.image_transform = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(*(memory_config.jitter,) * 4)],
                                   p=memory_config.jitter_prob), self.to_tensor])
        self.depth_image_transform = transforms.ToTensor()
        self.batch_size = None

    def add_batch(self, batch):
        if global_config.superpixels:
            images, targets, superpixels = batch
        else:
            images, targets = batch
            superpixels = [None] * len(images)

        for image, target, superpixel in zip(images, targets, superpixels):
            if self.last == self.config.capacity:
                self.last = 0

            if global_config.depth:
                image, depth = image
                self.images[self.last] = self.to_pil(image)
                self.depths[self.last] = self.to_pil(depth)
            else:
                self.images[self.last] = self.to_pil(image)

            self.targets[self.last] = target

            if global_config.superpixels:
                self.superpixels[self.last] = superpixel

            if self.prioritized_replay:
                self.priorities[self.last] = self.config.initial_priority

            self.last += 1
            self.max = max(self.max, self.last)

    def initialize_loader(self, batch_size):
        self.batch_size = batch_size

    def iterator(self):
        probs = self.priorities[:self.max] / self.priorities[:self.max].sum() if self.prioritized_replay else None
        index_batches = np.random.choice(self.max, self.max // self.batch_size * self.batch_size,
                                         replace=False, p=probs).reshape((-1, self.batch_size))
        return (self.make_batch(indices) for indices in index_batches)

    def make_batch(self, indices):
        images, targets = [], []
        superpixels = [] if global_config.superpixels else None
        for i in indices:
            image = (self.images[i], self.depths[i]) if global_config.depth else self.images[i]

            image, target, superpixel = self.flip(image, self.targets[i],
                                                  self.superpixels[i] if global_config.superpixels else None)

            if global_config.depth:
                image = torch.cat([self.image_transform(image[0]).float(),
                                   self.depth_image_transform(image[1])], dim=0)
            else:
                image = self.image_transform(image).float()

            images.append(image)

            target = [torch.from_numpy(t) for t in target]
            targets.append(target)

            if global_config.superpixels:
                superpixels.append(torch.from_numpy(superpixel))

        images = torch.stack(images)
        targets = (list(x) for x in zip(*targets))
        targets = tuple(torch.stack(x) for x in targets)

        batch = dict(images=images, targets=targets)

        if global_config.superpixels:
            batch['superpixels'] = torch.stack(superpixels)

        if self.prioritized_replay:
            batch['indices'] = indices
        if self.prioritized_replay and self.config.bias_correct:
            batch['weights'] = torch.tensor([1/self.priorities[i] for i in indices])
        else:
            batch['weights'] = torch.ones(len(indices))

        return batch

    def flip(self, image, target, superpixel):
        if np.random.random(1) < self.config.flip_prob:
            if global_config.depth:
                image = (tf.hflip(image[0]), tf.hflip(image[1]))
            else:
                image = tf.hflip(image)

            target = tuple(np.flip(t, axis=-1).copy() for t in target)

            if global_config.superpixels:
                superpixel = np.flip(superpixel, axis=-1).copy()

        return image, target, superpixel

    def load_memory(self, path):
        arrays = pickle.load(open(path, 'rb'))

        images, targets = arrays[0][-self.config.capacity:], arrays[1][-self.config.capacity:]
        size = len(images)
        self.images[:size] = images
        self.targets[:size] = targets

        if global_config.depth:
            depths = arrays[2][-self.config.capacity:]
            self.depths[:size] = depths
        if global_config.superpixels:
            superpixels = arrays[3][-self.config.capacity:]
            self.superpixels[:size] = superpixels

        self.max = size
        self.last = size

    def save_memory(self, path):
        arrays = (self.images[:self.max], self.targets[:self.max])
        if global_config.depth:
            arrays = arrays + (self.depths[:self.max],)
        if global_config.superpixels:
            arrays = arrays + (self.superpixels[:self.max],)
        file = open(path, 'wb')
        pickle.dump(arrays, file)
        file.close()

    def base_image_transform(self, image):
        if global_config.depth:
            image = tuple(self.to_tensor(self.to_pil(im)) for im in image)
        else:
            image = self.to_tensor(self.to_pil(image))

        if global_config.depth:
            image = torch.cat(list(image), dim=0)

        return image

    def update_priorities(self, batch_indices: list, priorities: list):
        for index, priority in zip(batch_indices, priorities):
            self.priorities[index] = priority
