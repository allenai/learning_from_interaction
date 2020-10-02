import torch
from torch import nn
import numpy as np
from skimage.segmentation import felzenszwalb
from skimage.measure import label
from torchvision import transforms

from config import ModelConfigFgBg, global_config
from models.backbones import UNetBackbone
from models.model import Model


class FgBgModel(Model):
    """
    This is a foreground-background segmentation model rewired to propose instances by post-processing the foreground
    with superpixels and extracting connected components.

    It is not designed to be used as an active model (in the sense that its compute_action) is inefficient.
    """
    def __init__(self, model_config: ModelConfigFgBg):
        super(FgBgModel, self).__init__()
        assert model_config.uncertainty is False, 'This is a passive model'
        self.config = model_config
        self.backbone = UNetBackbone(model_config)
        self.head = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1), nn.ReLU(),
                                  nn.Conv2d(128, 1 + self.config.uncertainty, 1))
        self.poking_grid = [(i, j) for i in range(global_config.grid_size) for j in range(global_config.grid_size)]
        self.to_pil = transforms.ToPILImage()

    def forward(self, images: torch.tensor, targets=None):
        y, _ = self.backbone(images)
        y = self.head(y)
        return y, None

    def compute_actions(self, images: torch.tensor, num_pokes: int, episode: int, episodes: int):
        with torch.no_grad():
            out, _ = self.forward(images)
            objectness = out[:, 0].sigmoid().cpu().numpy()

        x = [self.connected_components(mask, scores)
             for mask, scores in zip(objectness > .5, objectness)]

        actions = [[z[1] for z in y] for y in x]

        return 'this should never have run', actions, (out,)

    def compute_masks(self, images: torch.tensor, threshold: float):
        if type(images) == tuple and self.config.superpixel:
            images, superpixels = images[0], images[1]
        else:
            superpixels = self.compute_superpixels(images)

        with torch.no_grad():
            out, _ = self.forward(images)
            objectness = out[:, 0].sigmoid().cpu().numpy()

        x = [self.connected_components(mask, scores)
             for mask, scores in zip(objectness > threshold, objectness)]

        pred_masks = [[z[0] for z in y] for y in x]
        actions = [[z[1] for z in y] for y in x]
        pred_scores = [[z[2] for z in y] for y in x]

        pred_masks = [[self.postprocess_mask_with_sp(mask, superpixels) for mask in masks]
                      for masks, superpixels in zip(pred_masks, superpixels)]

        return actions, pred_masks, (out,), pred_scores

    def connected_components(self, mask, scores):
        fat_mask = self.fatten(mask) if self.config.fatten else mask
        labels = label(fat_mask) * mask
        labels[labels == 0] = -1

        mps = []
        for i in np.unique(labels):
            if i == -1:
                continue
            m = labels == i
            if m.sum() > 5:
                s = float((m * scores).max())
                p = (m * scores).argmax()
                p = dict(point=(p // global_config.grid_size, p % global_config.grid_size))
                mps.append((m, p, s))
        if len(mps) > 0:
            mps = sorted(mps, key=lambda x: x[-1], reverse=True)
        return mps

    def compute_superpixels(self, images):
        images = [np.array(self.to_pil(image)) for image in images[:, :3].cpu()]
        return [felzenszwalb(image, scale=200, sigma=.5, min_size=200)[::global_config.stride,
                                                                       ::global_config.stride].astype(np.int32)
                for image in images]

    @staticmethod
    def postprocess_mask_with_sp(mask, superpixels):
        smoothed_mask = np.zeros_like(mask)
        superpixels = [superpixels == i for i in np.unique(superpixels)]
        for superpixel in superpixels:
            if mask[superpixel].sum() / superpixel.sum() > .25:
                smoothed_mask[superpixel] = True
        return smoothed_mask

    @staticmethod
    def fatten(mask):
        fat_mask = mask.copy()
        fat_mask[:-1] = fat_mask[:-1] | mask[1:]
        fat_mask[1:] = fat_mask[1:] | mask[:-1]
        fat_mask[:, :-1] = fat_mask[:, :-1] | mask[:, 1:]
        fat_mask[:, 1:] = fat_mask[:, 1:] | mask[:, :-1]
        return fat_mask
