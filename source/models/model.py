import torch
from torch import nn


class Model(nn.Module):
    """
    Abstract Model class for instance segmentation.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, images: torch.tensor, *targets):
        """
        As usual, this function feeds directly into the loss function, from which it also receives its gradients.
        The targets are an optional input, used for example in multi stage detectors with "teacher forcing" style
        training.
        """
        raise NotImplementedError

    def compute_actions(self, images: torch.tensor, num_pokes: int, episode: int, episodes: int):
        """
        This is the function through which the model is used during data collection. It is run with the model in
        eval mode, and no gradients are computed. The function should return num_pokes poking locations and forces for
        each image. It should also return the output of the forward pass, from which statistics can be computed to
        monitor the training.
        """
        raise NotImplementedError

    def compute_masks(self, images: torch.tensor, threshold: float):
        """
        This is the function through which the model performs inference. It is run in eval mode and no gradients are
        computed. For each image, it should predict poking locations of all objects, segmentation masks and relative
        masses for these objects, and confidence scores for these proposals. It should also return the output of the
        forward pass.
        """
        raise NotImplementedError


