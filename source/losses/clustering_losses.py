import numpy as np
import torch

from config import ObjectnessClusteringLossConfig, MaskAndMassLossConfig, global_config
from losses.loss_utils import focal_loss_function, embedding_loss_function, MassLoss
from losses.losses import LossFunction, ObjectnessLoss


class MaskAndMassLoss(LossFunction):
    """
    This loss assumes that the model predicts: pixel feature embeddings, pixel objectness logits, pixel force logits

    The loss encourages the model to achieve the following
        -   Pixel objectness logits are large for pixels that are likely to move when poked.
        -   Force logits are non-random at locations of high objectness. At these locations, the largest of the logits
            corresponds to a force that moves the object, but so that no lower force will move the object
        -   Pixel feature embeddings are close to each other for pixels that are likely to move together after a poke

    Note: Terminology "Force" vs "Mass" is somewhat ambiguous in this class. Both refer to the same.
    """

    def __init__(self, loss_config: MaskAndMassLossConfig):
        super(MaskAndMassLoss, self).__init__()
        self.loss_summary_length = 2 + global_config.superpixels + loss_config.filter
        self.prioritized_replay = loss_config.prioritized_replay
        self.config = loss_config
        self.focal_loss = focal_loss_function(loss_config.objectness_weight)
        self.embedding_loss = embedding_loss_function(loss_config.threshold)
        self.mass_loss = MassLoss.apply
        self.logsm = torch.nn.LogSoftmax(dim=0)
        self.distance_function = lambda x, y: ((x - y) ** 2).sum(dim=0)

    def __call__(self, model_predictions, targets, weights, superpixels=None):
        objectnesss, masses, embeddings = model_predictions[:3]
        device = embeddings[0].device

        object_masks, foreground_masks, background_masks, mass_masks = targets

        embedding_losses, objectness_losses, mass_losses = [], [], []

        for weight, embedding, objectness, mass, object_mask, foreground, background, mass_mask in zip(
                weights, embeddings, objectnesss, masses, object_masks, foreground_masks, background_masks, mass_masks):

            embedding_loss = self.compute_embedding_loss(embedding, object_mask, weight, device)
            if not self.config.instance_only:
                if self.config.scaleable:
                    mass_loss = self.mass_loss(mass, mass_mask, weight)
                else:
                    mass_loss = self.mass_loss_nonscaleable(mass, mass_mask, weight)
                mass_losses.append(mass_loss * self.config.mass_loss_weight)

            objectness_loss = self.compute_objectness_loss(objectness, foreground, background, weight)

            embedding_losses.append(embedding_loss)
            objectness_losses.append(objectness_loss)

        losses = [embedding_losses, objectness_losses] + ([mass_losses] if not self.config.instance_only else [])
        losses = self.stack_losses(losses, device)

        if self.prioritized_replay:
            priorities = self.compute_priorities([embedding_losses])
            return losses, priorities

        return losses

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

    def compute_embedding_loss(self, embedding, object_mask, weight, device):
        max_objects = object_mask.shape[0]
        embedding_loss = torch.tensor(0., dtype=torch.float32).to(device)

        objs = [object_mask[i] for i in range(max_objects) if object_mask[i].sum() > 0]
        for obj in objs:
            center = embedding[:, obj].mean(1).unsqueeze(1).unsqueeze(2)
            embedding_loss = embedding_loss + self.embedding_loss(self.distance_function(embedding, center),
                                                                  obj, weight)
        return embedding_loss / max(len(objs), 1)

    def mass_loss_nonscaleable(self, mass, mass_mask, weight):
        mass = self.logsm(mass)
        classes = mass_mask.argmax(dim=0)
        pointwise_ce = sum(mass[i] * (classes == i) for i in range(3))
        loss = (pointwise_ce * mass_mask.sum(dim=0)).sum(dim=(0, 1))
        return - loss * weight

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
        mass_mask = np.zeros((3, global_config.grid_size, global_config.grid_size), dtype=np.float32)
        background_mask = np.zeros((global_config.grid_size, global_config.grid_size), dtype=np.float32)
        successes = 0

        for i, (action, feedback) in enumerate(zip(actions, feedbacks)):
            if self.config.instance_only:
                mask, mass_fb = feedback, None
            else:
                mask, mass_fb = feedback[0], feedback[1]
            action, mass = action['point'], action['force'] if not self.config.instance_only else None
            weights = self.get_neighbourhood(action, superpixel)
            score = self.get_score(mask, action)
            if score > self.config.foreground_threshold:
                object_masks[i, mask] = True
                foreground_mask += weights
                if not self.config.instance_only:
                    if self.config.scaleable:
                        mass_mask += self.mass_feedback_vector(mass, mass_fb) * weights[None, ...]
                    else:
                        mass_mask += self.mass_feedback_vector_nonscaleable(mass, mass_fb) * weights[None, ...]
                successes += 1
            else:
                background_mask += weights
        return (object_masks, foreground_mask, background_mask, mass_mask), successes

    @staticmethod
    def mass_feedback_vector(mass, feedback):
        vec = np.zeros(3, dtype=np.float32)
        if feedback == 2:
            return vec[:, None, None]
        if feedback == 0:
            vec[mass] = 1
        elif feedback == -1:
            for i in range(0, mass):
                vec[i] = 1
        elif feedback == 1:
            for i in range(mass + 1, 3):
                vec[i] = 1
        vec = vec - vec.mean()
        vec = vec / np.abs(vec).sum()
        return vec[:, None, None]

    @staticmethod
    def mass_feedback_vector_nonscaleable(mass, feedback):
        vec = np.zeros(3, dtype=np.float32)
        if feedback < 2:
            vec[feedback] = 1
        return vec[:, None, None]

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

    @staticmethod
    def stack_losses(losses, device):
        losses = [torch.stack(l) if len(l) > 0 else torch.tensor(0.).to(device)
                  for l in losses]
        losses = [l.sum() / (l > 0).sum().clamp(min=1) for l in losses]
        return losses

    def compute_priorities(self, losses: list):
        priorities = []
        for loss in zip(*losses):
            score = min(iou.item() if iou > 0 else .5 for iou in loss)
            priorities.append((score - .5) ** 2 + .02)
        return priorities


class ObjectnessClusteringLoss(ObjectnessLoss):
    """
    This loss is for instance segmentation only.
    It assumes that the model predicts: pixel feature embeddings, pixel objectness logits,
    (and optionally a single filter logit for the entire image).

    The losses for objectness and filter logit are described in the parent class.
        -   Pixel feature embeddings are close to each other for pixels that are likely to move together after a poke
    """

    def __init__(self, loss_config: ObjectnessClusteringLossConfig):
        super(ObjectnessClusteringLoss, self).__init__(loss_config)
        self.loss_summary_length = 2 + global_config.superpixels + loss_config.filter

        self.embedding_loss = embedding_loss_function(loss_config.threshold)
        self.distance_function = lambda x, y: ((x - y) ** 2).sum(dim=0)

    def __call__(self, model_predictions, targets, weights, superpixels=None):
        objectnesss, embeddings = model_predictions[:2]
        device = embeddings[0].device
        if self.config.filter:
            filter_logits = model_predictions[-1]
        else:
            filter_logits = [None] * embeddings.shape[0]

        object_masks, foreground_masks, background_masks = targets

        filter_and_smoothness_losses = \
            self.compute_filter_and_smoothness_loss(filter_logits, model_predictions[:2], object_masks,
                                                    weights, superpixels)

        embedding_losses, objectness_losses = [], []

        for weight, embedding, objectness, filter_logit, object_mask, foreground, background in zip(
                weights, embeddings, objectnesss, filter_logits, object_masks, foreground_masks, background_masks):

            if self.config.filter and filter_logit < self.config.filter_threshold:
                continue

            embedding_loss = self.compute_embedding_loss(embedding, object_mask, weight, device)

            if self.config.center_foreground:
                foreground = self.center_foreground(foreground, embedding)
            objectness_loss = self.compute_objectness_loss(objectness, foreground, background, weight)

            embedding_losses.append(embedding_loss)
            objectness_losses.append(objectness_loss)

        losses = self.stack_losses([embedding_losses, objectness_losses], device)

        losses += filter_and_smoothness_losses

        if self.prioritized_replay:
            priorities = self.compute_priorities([embedding_losses])
            return losses, priorities

        return losses

    def compute_embedding_loss(self, embedding, object_mask, weight, device):
        max_objects = object_mask.shape[0]
        embedding_loss = torch.tensor(0., dtype=torch.float32).to(device)

        objs = [object_mask[i] for i in range(max_objects) if object_mask[i].sum() > 0]
        for obj in objs:
            if self.config.robustify is not None:
                obj = self.robustify(embedding, obj)
            center = embedding[:, obj].mean(1).unsqueeze(1).unsqueeze(2)
            embedding_loss = embedding_loss + self.embedding_loss(self.distance_function(embedding, center),
                                                                  obj, weight)
        return embedding_loss / max(len(objs), 1)

    @staticmethod
    def center_foreground(foreground, embedding):
        if torch.any(foreground > .999):
            with torch.no_grad():
                emb = embedding.transpose(0, 1).transpose(1, 2)
                feats = emb[torch.where(foreground > .999)]
                masks = ((embedding.unsqueeze(0) - feats.unsqueeze(2).unsqueeze(3)) ** 2).sum(
                    dim=1) < 1
                meanfeats = torch.stack([emb[mask].mean(0) for mask in masks])
                meanmasks = ((embedding.unsqueeze(0) - meanfeats.unsqueeze(2).unsqueeze(3)) ** 2).sum(dim=1) < 1
                ious = (masks * meanmasks).sum(dim=(1, 2)).float() / (masks | meanmasks).sum(dim=(1, 2)) > .8
                foreground[torch.where(foreground > .999)] += .2 * (2 * ious - 1).float()
        return foreground.clone()

    def robustify(self, embedding, obj):
        with torch.no_grad():
            center = embedding[:, obj].mean(1).unsqueeze(1).unsqueeze(2)
            distances = self.distance_function(embedding, center)
            throw_out = (distances > 1 - self.config.robustify[0]) * obj
            put_in = (distances < 1 + self.config.robustify[1]) * (~ obj)
            return (obj * (~ throw_out)) | put_in


class ObjectnessClusteringLossGT(ObjectnessClusteringLoss):
    """
    This loss is an alternative for clustering based instance segmentation models trained fully supervised.
    """

    def __init__(self, loss_config: ObjectnessClusteringLossConfig):
        super(ObjectnessClusteringLossGT, self).__init__(loss_config)

    def __call__(self, model_predictions, targets, weights, superpixels=None):
        seed_scores, embeddings = model_predictions[:2]
        device = embeddings[0].device

        object_masks, _, _ = targets

        embedding_losses, seed_losses = [], []

        for weight, embedding, seed_score, object_mask in zip(
                weights, embeddings, seed_scores, object_masks):
            embedding_loss = self.compute_embedding_loss(embedding, object_mask, weight, device)

            seed_loss = self.compute_seed_loss(seed_score, embedding, object_mask, weight)

            embedding_losses.append(embedding_loss)
            seed_losses.append(seed_loss)

        losses = self.stack_losses([embedding_losses, seed_losses], device)

        if self.prioritized_replay:
            priorities = self.compute_priorities([embedding_losses])
            return losses, priorities

        return losses

    def compute_seed_loss(self, seed_score, embedding, object_mask, weight):
        foreground, background = self.make_seed_targets(embedding, object_mask, seed_score)
        return self.focal_loss(seed_score, foreground, background, weight)

    def make_seed_targets(self, embedding, masks, seed_score):
        with torch.no_grad():
            foreground, background = torch.zeros_like(seed_score), torch.zeros_like(seed_score)
            dim = len(foreground.view(-1))
            masks = [mask for mask in masks if mask.sum() > 0]
            for mask in masks:
                center = embedding[:, mask].mean(1).unsqueeze(1).unsqueeze(2)
                distances = self.distance_function(embedding, center)
                dmin = distances.view(-1).kthvalue(6)[0]
                dmax = distances.view(-1).kthvalue(dim - 5)[0]
                foreground += distances < dmin
                background += distances > dmax
            return foreground, background


class PooledMaskAndMassLoss(LossFunction):
    """
    This loss is the equivalent of MaskAndMassLoss for models that do not produce pixel-wise force logits,
    but instance-wise force logits
    """

    def __init__(self, loss_config: MaskAndMassLossConfig):
        super(PooledMaskAndMassLoss, self).__init__()
        self.loss_summary_length = 2 + global_config.superpixels + loss_config.filter
        self.prioritized_replay = loss_config.prioritized_replay
        self.config = loss_config
        self.focal_loss = focal_loss_function(loss_config.objectness_weight)
        self.embedding_loss = embedding_loss_function(loss_config.threshold)
        self.mass_loss = MassLoss.apply
        self.logsm = torch.nn.LogSoftmax(dim=1)
        self.distance_function = lambda x, y: ((x - y) ** 2).sum(dim=0)

    def mass_loss_nonscaleable(self, mass_logit, mass_target, weight):
        mass = self.logsm(mass_logit)
        classes = mass_target.argmax(dim=1)
        pointwise_ce = sum(mass[:, i] * (classes == i) for i in range(3))
        loss = (pointwise_ce * mass_target.sum(dim=1)).sum()
        return - loss * weight

    def __call__(self, model_predictions, targets, weights, superpixels=None):
        objectnesss, embeddings, mass_logits, _ = model_predictions
        device = embeddings[0].device

        object_masks, foreground_masks, background_masks, mass_targets = targets

        embedding_losses, objectness_losses, mass_losses = [], [], []

        for weight, embedding, objectness, mass_logit, object_mask, foreground, background, mass_target in zip(
                weights, embeddings, objectnesss, mass_logits, object_masks, foreground_masks, background_masks,
                mass_targets):

            embedding_loss = self.compute_embedding_loss(embedding, object_mask, weight, device)
            if self.config.scaleable:
                mass_loss = self.mass_loss(mass_logit, mass_target, weight)
            else:
                mass_loss = self.mass_loss_nonscaleable(mass_logit, mass_target, weight)
            objectness_loss = self.compute_objectness_loss(objectness, foreground, background, weight)

            embedding_losses.append(embedding_loss)
            mass_losses.append(mass_loss)
            objectness_losses.append(objectness_loss)

        losses = self.stack_losses([embedding_losses, objectness_losses, mass_losses], device)

        if self.prioritized_replay:
            priorities = self.compute_priorities([embedding_losses])
            return losses, priorities

        return losses

    def compute_embedding_loss(self, embedding, object_mask, weight, device):
        max_objects = object_mask.shape[0]
        embedding_loss = torch.tensor(0., dtype=torch.float32).to(device)

        objs = [object_mask[i] for i in range(max_objects) if object_mask[i].sum() > 0]
        for obj in objs:
            center = embedding[:, obj].mean(1).unsqueeze(1).unsqueeze(2)
            embedding_loss = embedding_loss + self.embedding_loss(self.distance_function(embedding, center),
                                                                  obj, weight)
        return embedding_loss / max(len(objs), 1)

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
        mass_targets = np.zeros((global_config.max_pokes, 3), dtype=np.float32)
        background_mask = np.zeros((global_config.grid_size, global_config.grid_size), dtype=np.float32)
        successes = 0

        for i, (action_and_mass, mask_and_mass) in enumerate(zip(actions, feedbacks)):
            mask, mass_fb = mask_and_mass
            action, mass = action_and_mass['point'], action_and_mass['force']
            weights = self.get_neighbourhood(action, superpixel)
            score = self.get_score(mask, action)
            if score > self.config.foreground_threshold:
                object_masks[i, mask] = True
                foreground_mask += weights
                if self.config.scaleable:
                    mass_targets[i] = self.mass_feedback_vector(mass, mass_fb)
                else:
                    mass_targets[i] = self.mass_feedback_vector_nonscaleable(mass, mass_fb)
                successes += 1
            else:
                background_mask += weights
        return (object_masks, foreground_mask, background_mask, mass_targets), successes

    @staticmethod
    def mass_feedback_vector(mass, feedback):
        vec = np.zeros(3, dtype=np.float32)
        if feedback == 2:
            return vec
        if feedback == 0:
            vec[mass] = 1
        elif feedback == -1:
            for i in range(0, mass):
                vec[i] = 1
        elif feedback == 1:
            for i in range(mass + 1, 3):
                vec[i] = 1
        vec = vec - vec.mean()
        vec = vec / np.abs(vec).sum()
        return vec

    @staticmethod
    def mass_feedback_vector_nonscaleable(mass, feedback):
        vec = np.zeros(3, dtype=np.float32)
        if feedback < 2:
            vec[feedback] = 1
        return vec

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
            score = min(iou.item() if iou > 0 else .5 for iou in loss)
            priorities.append((score - .5) ** 2 + .02)
        return priorities
