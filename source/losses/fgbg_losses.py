import numpy as np
import torch

from config import FgBgLossConfig, global_config
from losses.loss_utils import focal_loss_function
from losses.losses import LossFunction


class FgBgLossFunction(LossFunction):
    """
    This is a simple loss for foreground-background segmentation models with some adjustability for unlabeled pixels.
    """
    def __init__(self, loss_config: FgBgLossConfig):
        super(FgBgLossFunction, self).__init__()
        self.loss_summary_length = 1
        self.prioritized_replay = loss_config.prioritized_replay
        self.config = loss_config
        self.focal_loss = focal_loss_function(1)

    def __call__(self, model_predictions: tuple, targets: tuple, weights, *superpixels):
        _, foreground_masks, background_masks = targets
        objectness_losses = []
        for weight, objectness, foreground, background in zip(
                weights, model_predictions[0], foreground_masks, background_masks):
            objectness_loss = self.compute_objectness_loss(objectness, foreground, background, weight)
            objectness_losses.append(objectness_loss)
        losses = [torch.stack(objectness_losses)]

        if self.prioritized_replay:
            priorities = self.compute_priorities(losses)
            return [l.sum() / (l > 0).sum().clamp(min=1) for l in losses], priorities

        return [l.sum() / (l > 0).sum().clamp(min=1) for l in losses]

    def process_feedback(self, actions: list, feedback: list, superpixels=None):
        targets = []
        num_successes = 0
        for act, fb in zip(actions, feedback):
            target, new_successes = self.process_single_feedback(act, fb)
            targets.append(target)
            num_successes += new_successes
        return targets, num_successes

    def process_single_feedback(self, actions, feedbacks):
        foreground_mask = np.zeros((global_config.grid_size, global_config.grid_size), dtype=np.float32)
        background_mask = np.zeros((global_config.grid_size, global_config.grid_size), dtype=np.float32)
        poking_mask = np.zeros((global_config.max_pokes, global_config.grid_size, global_config.grid_size),
                               dtype=np.bool)
        successes = 0

        for i, (action, mask, pm) in enumerate(zip(actions, feedbacks, poking_mask)):
            weights = self.get_neighbourhood(action['point'])
            if mask.sum() > self.config.foreground_threshold:
                if self.config.restrict_positives:
                    foreground_mask += weights
                    pm[:] = mask > 0
                else:
                    foreground_mask = (foreground_mask + mask) > 0
                successes += 1
            elif self.config.restrict_negatives:
                background_mask += weights
        if not self.config.restrict_negatives:
            background_mask = ~ foreground_mask
        return (poking_mask, foreground_mask, background_mask), successes

    def get_neighbourhood(self, action):
        x, y = action
        weights = np.zeros((global_config.grid_size, global_config.grid_size), dtype=np.float32)
        dx1 = min(x, self.config.kernel_size)
        dx2 = min(global_config.grid_size - 1 - x, self.config.kernel_size) + 1
        dy1 = min(y, self.config.kernel_size)
        dy2 = min(global_config.grid_size - 1 - y, self.config.kernel_size) + 1
        x1, x2, y1, y2 = x - dx1, x + dx2, y - dy1, y + dy2
        weights[x1:x2, y1:y2] = self.config.kernel[self.config.kernel_size - dx1:
                                                   self.config.kernel_size + dx2,
                                                   self.config.kernel_size - dy1:
                                                   self.config.kernel_size + dy2]
        return weights

    def compute_objectness_loss(self, objectness, foreground, background, weight):
        b = objectness.shape[0] > 1
        objectness, uncertainty = objectness[0], objectness[1] if b else None
        objectness_loss = self.focal_loss(objectness, foreground, background, weight)
        if b:
            uncertainty_foreground = foreground * (objectness < 0) + background * (objectness > 0)
            uncertainty_background = foreground * (objectness >= 0) + background * (objectness <= 0)
            uncertainty_loss = self.focal_loss(uncertainty, uncertainty_foreground, uncertainty_background, weight)
            return objectness_loss + uncertainty_loss
        return objectness_loss

    def compute_priorities(self, losses: list):
        raise NotImplementedError


class SoftMaskLossFunction(LossFunction):
    """
    This is an L2 loss for fitting soft fg-bg targets. It is used for the videoPCA baseline.
    """
    def __init__(self):
        super(SoftMaskLossFunction, self).__init__()
        self.loss_summary_length = 2

    def __call__(self, model_predictions: tuple, targets: tuple, weights, *superpixels):
        soft_masks = targets[0]
        objectness_losses = []
        for weight, objectness, soft_mask in zip(weights, model_predictions[0].sigmoid(), soft_masks):
            loss = weight * ((objectness - soft_mask)**2).sum()
            objectness_losses.append(loss)
        losses = [torch.stack(objectness_losses)]

        return [l.sum() / (l > 0).sum().clamp(min=1) for l in losses]

    def process_feedback(self, actions: list, feedback: list, superpixels=None):
        targets = []
        num_successes = 0
        for act, fb in zip(actions, feedback):
            target, new_successes = self.process_single_feedback(act, fb)
            targets.append(target)
            num_successes += new_successes
        return targets, num_successes

    @staticmethod
    def process_single_feedback(actions, feedbacks):
        soft_mask = np.zeros((global_config.grid_size, global_config.grid_size), dtype=np.float32)
        successes = 0

        for i, (action, mask) in enumerate(zip(actions, feedbacks)):
            soft_mask += mask
            successes += mask.sum() > 0
        return (soft_mask, ), successes

    def compute_priorities(self, losses: list):
        raise NotImplementedError
