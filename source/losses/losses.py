import numpy as np
import torch

from config import ObjectnessLossConfig, global_config
from losses.loss_utils import focal_loss_function, SmoothnessPenalty


class LossFunction:
    def __init__(self):
        self.loss_summary_length = 0  # the length of the output of the loss function
        self.prioritized_replay = False

    def __call__(self, model_predictions: tuple, targets: tuple, weights, *superpixels):
        """
        :param model_predictions: The output of the model's forward
        :param targets: The targets supplied by the memory's iterator
        :param weights: The weights for weighting each datapoint's gradient (for bias reduction in prioritized replay)
        :param superpixels: Optionally, the list of superpixels for the images corresponding to the model_predictions.
        :return: The loss, that is one or several scalars
        """
        raise NotImplementedError

    def process_feedback(self, actions: list, feedback: list, *superpixels: list):
        """
        :param actions: for each data point, a list of poking locations used in its collection
        :param feedback: for each data point, the feedback received for each of the poking locations
        :param superpixels: Optionally, the superpixels corresponding to each data point
        :return: The targets for each data point, in a format ready to be added to the memory

        This function post-processes the feedback obtained from the Actors, in a format ready to be added to the memory.
        """
        raise NotImplementedError

    def compute_priorities(self, losses: list):
        """
        :param losses: list of losses for each data point
        :return: list of priorities for each data point, to be used by the replay memory for optional prioritized replay
        """
        raise NotImplementedError


class ObjectnessLoss(LossFunction):
    """
    A superclass that implements certain utilities used by models that learn an objectness score for greedy poking.
    It is not a stand alone loss, and does not implement the __call__ method. It does implement the process_feedback
    method.

    The loss encourages the model to achieve the following
        -  Pixel objectness logits are large for pixels that are likely to move when poked.
        -  Optionally, features are encouraged to be constant along a superpixel.
        -  An optional filter logit is large when an image is likely to contain easy to poke objects.

    If the filter logits are used, images whose filter logit is too negative will not be used to learn
    objectness or features used in instance mask prediction.
    """

    def __init__(self, loss_config: ObjectnessLossConfig):
        super(ObjectnessLoss, self).__init__()
        self.prioritized_replay = loss_config.prioritized_replay
        self.config = loss_config
        self.focal_loss = focal_loss_function(loss_config.objectness_weight)
        if global_config.superpixels and loss_config.smoothness_weight > 0:
            self.smoothness_loss = SmoothnessPenalty()

    def __call__(self, model_predictions: tuple, targets: tuple, weights: list, *superpixels):
        raise NotImplementedError

    def process_feedback(self, actions: list, feedback: list, superpixels=None):
        targets = []
        num_successes = 0
        if not superpixels:
            superpixels = [None] * len(actions)
        for act, fb, superpixel in zip(actions, feedback, superpixels):
            target, new_successes = self.process_single_feedback(act, fb, superpixel)
            targets.append(target)
            num_successes += new_successes
        return targets, num_successes

    def process_single_feedback(self, actions, feedbacks, superpixel):
        object_masks = np.zeros((global_config.max_pokes,
                                 global_config.grid_size, global_config.grid_size), dtype=bool)
        foreground_mask = np.zeros((global_config.grid_size, global_config.grid_size), dtype=np.float32)
        background_mask = np.zeros((global_config.grid_size, global_config.grid_size), dtype=np.float32)
        successes = 0

        for i, (action, mask) in enumerate(zip(actions, feedbacks)):
            weights = self.get_neighbourhood(action['point'], superpixel)
            score = self.get_score(mask, action['point'])
            if score > self.config.foreground_threshold:
                object_masks[i, mask] = True
                foreground_mask += weights
                successes += 1
            else:
                background_mask += weights
        return (object_masks, foreground_mask, background_mask), successes

    def get_score(self, mask, action):
        if not self.config.localize_object_around_poking_point:
            return mask.sum()
        x, y = action
        dx1 = min(x, self.config.kernel_size)
        dx2 = min(global_config.grid_size - 1 - x, self.config.kernel_size) + 1
        dy1 = min(y, self.config.kernel_size)
        dy2 = min(global_config.grid_size - 1 - y, self.config.kernel_size) + 1
        x1, x2, y1, y2 = x - dx1, x + dx2, y - dy1, y + dy2
        return (mask[x1:x2, y1:y2] * self.config.check_change_kernel[self.config.kernel_size - dx1:
                                                                     self.config.kernel_size + dx2,
                                                                     self.config.kernel_size - dy1:
                                                                     self.config.kernel_size + dy2]).sum()

    def get_neighbourhood(self, action, superpixel):
        x, y = action
        if self.config.point_feedback_for_action:
            weights = np.zeros((global_config.grid_size, global_config.grid_size), dtype=np.float32)
            weights[x, y] = 1
        elif superpixel is not None and self.config.superpixel_for_action_feedback:
            weights = (superpixel == superpixel[x, y]).astype(np.float32)
        else:
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

    def compute_filter_and_smoothness_loss(self, filter_logits, features_for_smoothing: tuple,
                                           object_masks, weights, superpixels):
        filter_and_smoothness_losses = []
        if self.config.filter:
            filter_logit_positives = (object_masks.sum(dim=(1, 2, 3)) > 0).float()
            filter_logit_negatives = 1 - filter_logit_positives
            filter_logits_losses = torch.stack([self.focal_loss(fl, flp, fln, weight) for weight, fl, flp, fln
                                                in zip(weights, filter_logits, filter_logit_positives,
                                                       filter_logit_negatives)])
            filter_and_smoothness_losses.append(filter_logits_losses.sum())

        if superpixels is not None and self.config.smoothness_weight > 0:
            smoothness_losses = []
            features_for_smoothing = list(zip(*features_for_smoothing))

            for weight, features, superpixel in zip(weights, features_for_smoothing, superpixels):
                smoothness_loss = sum(self.smoothness_loss(feature, superpixel) for feature in features)
                smoothness_losses.append(smoothness_loss * self.config.smoothness_weight * weight)
            filter_and_smoothness_losses.append(torch.stack(smoothness_losses).sum())

        return filter_and_smoothness_losses

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

    @staticmethod
    def stack_losses(losses, device):
        losses = [torch.stack(l) if len(l) > 0 else torch.tensor(0.).to(device) for l in losses]
        losses = [l.sum() / (l > 0).sum().clamp(min=1) for l in losses]
        return losses

    def compute_priorities(self, losses: list):
        priorities = []
        for loss in zip(*losses):
            score = min(iou.item() if iou > 0 else self.config.prioritize_default for iou in loss)
            priorities.append(self.config.prioritize_function(score))
        return priorities
