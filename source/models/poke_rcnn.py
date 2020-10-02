import torch
import numpy as np
from torch import nn
from random import sample
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Instances, Boxes, BitMasks
from detectron2.utils.events import EventStorage

from losses.losses import ObjectnessLossConfig, ObjectnessLoss
from losses.loss_utils import focal_loss_function
from models.model import Model
from models.backbones import make_rpn50_fpn_config
from config import global_config


class PokeRCNN(Model):
    """
    This wraps a standard detectron2 MaskRCNN (including standard losses) for instance segmentation, but also predicts
    objectness logits like the clustering models, and can therefore be used for fully self-supervised training.
    """

    def __init__(self, uncertainty=False):
        super(PokeRCNN, self).__init__()
        self.mask_rcnn = MaskRCNNWithPokeHead(uncertainty)
        self.poking_grid = [(i, j) for i in range(global_config.grid_size) for j in range(global_config.grid_size)]
        self.register_buffer('background_mask', torch.zeros(1, 800, 800, dtype=torch.int))
        self.register_buffer('background_box', torch.tensor([[1, 1, 799, 799]]))

    def forward(self, images: torch.tensor, targets=None):
        batched_inputs = self.rescale_and_zip(images, targets)
        if targets is None:
            return self.mask_rcnn.inference(batched_inputs)
        return self.mask_rcnn(batched_inputs)

    def rescale_and_zip(self, images, targets=None):
        with torch.no_grad():
            if targets is None:
                targets = [None] * images.shape[0]
            else:
                targets = list(zip(*targets))
            batched_output = []
            for image, target in zip(images, targets):
                d = {"image": nn.functional.interpolate(image.unsqueeze(0), (800, 800), mode='bilinear').squeeze(0)}
                if target is not None:
                    masks, foreground, background = target
                    instances = self.scale_and_process_masks(masks)
                    d["instances"] = instances
                    d["poking_targets"] = torch.stack([foreground, background])
                batched_output.append(d)
            return batched_output

    def scale_and_process_masks(self, masks):
        device = masks.device
        dummy_mask = torch.zeros_like(masks[0]).unsqueeze(0)
        non_emptys = masks.sum(dim=(1, 2)) > 0
        non_empty = non_emptys.sum().item()
        masks = torch.cat([masks[non_emptys], dummy_mask], dim=0) if non_empty else dummy_mask
        masks = (nn.functional.interpolate(masks.float().unsqueeze(1),
                                           size=(800, 800)) > .5).squeeze(1)

        if non_empty:
            box_coordinates = [torch.where(mask) for mask in masks[:-1]]
            box_coordinates = torch.tensor([[x[1].min(), x[0].min(), x[1].max(), x[0].max()] for x in box_coordinates])
            box_coordinates = torch.cat([box_coordinates.to(device), self.background_box], dim=0)
        else:
            box_coordinates = self.background_box

        instances = Instances((800, 800))
        instances.gt_boxes = Boxes(box_coordinates)
        instances.gt_masks = BitMasks(masks)
        classes = torch.zeros(non_empty + 1, dtype=torch.int64)
        classes[-1] = 1
        instances.gt_classes = classes
        return instances.to(device)

    @staticmethod
    def select_largest_on_mask(mask, scores):
        mask = mask.reshape(global_config.grid_size, global_config.stride,
                            global_config.grid_size, global_config.stride).mean(axis=(1, 3)) > .5
        argmax = (mask * scores).argmax() if np.any(mask) else scores.argmax()
        return argmax // global_config.grid_size, argmax % global_config.grid_size

    def compute_actions(self, images: torch.tensor, num_pokes: int, episode: int, episodes: int):
        with torch.no_grad():
            results, poking_scores = self.forward(images)
            poking_scores_numpy = poking_scores[:, 0].sigmoid().cpu().numpy()
            detections = [result['instances'].pred_masks.cpu().numpy() for result in results]

            actions = []
            for masks, poking_score in zip(detections, poking_scores_numpy):
                action = []
                for mask in masks[:num_pokes // 2]:
                    point = self.select_largest_on_mask(mask, poking_score)
                    action.append(dict(point=point))
                action += sample(self.poking_grid, num_pokes - len(action))
                actions.append(action)

        return actions, (poking_scores,)

    def compute_masks(self, images: torch.tensor, threshold: float):
        ret_masks, scores, actions = [], [], []
        self.eval()
        with torch.no_grad():
            results, poking_scores = self.forward(images)
            poking_scores_numpy = poking_scores[:, 0].sigmoid().cpu().numpy()
            detections = [(result['instances'].pred_masks.cpu().numpy(), result['instances'].scores.cpu().numpy(),
                           result['instances'].pred_classes.cpu().numpy())
                          for result in results]

        for (masks, mask_scores, classes), poking_score in zip(detections, poking_scores_numpy):
            action, new_masks, new_scores = [], [], []
            for mask, mask_score, cl, _ in zip(masks, mask_scores, classes, [None] * global_config.max_pokes):
                if cl > 0:
                    continue
                if mask_score < threshold:
                    break
                point = self.select_largest_on_mask(mask, poking_score)
                action.append(dict(point=point))
                new_masks.append(mask)
                new_scores.append(mask_score)
            ret_masks.append(new_masks)
            scores.append(new_scores)
            actions.append(action)
        return actions, ret_masks, (poking_scores,), scores


class MaskRCNNWithPokeHead(GeneralizedRCNN):
    def __init__(self, uncertainty=True):
        super(MaskRCNNWithPokeHead, self).__init__(make_rpn50_fpn_config())
        self.poking_head = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
                                         nn.Conv2d(64, 2, kernel_size=1))

        self.poking_loss = MaskPokingLoss(uncertainty)
        self.event_storage = EventStorage()

    def forward(self, batched_inputs):
        with self.event_storage:
            images = self.preprocess_image(batched_inputs)
            gt_instances = [x["instances"] for x in batched_inputs]
            features = self.backbone(images.tensor)
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
            poking_scores = self.poking_head(features['p3'])
            poking_targets = torch.stack([x['poking_targets'] for x in batched_inputs])
            poking_losses = self.poking_loss(poking_scores, poking_targets)

            losses = list(detector_losses.values()) + list(proposal_losses.values()) + [poking_losses]
            return losses

    def inference(self, batched_inputs, *kwargs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, features, proposals, None)
        poking_scores = self.poking_head(features['p3'])
        return self.postprocess(results, batched_inputs), poking_scores

    @staticmethod
    def postprocess(instances, batched_inputs):
        processed_results = []
        for results_per_image, input_per_image in zip(instances, batched_inputs):
            r = detector_postprocess(results_per_image.to('cpu'), 300, 300)
            processed_results.append({"instances": r})
        return processed_results


class DummyObjectnessLoss(ObjectnessLoss):
    def __init__(self, conf: ObjectnessLossConfig):
        super(DummyObjectnessLoss, self).__init__(conf)
        assert conf.prioritized_replay is False
        self.loss_summary_length = 6
        # NOTE: Since per-image mask losses are not accessible in MaskRCNN, prioritized replay is not supported.

    def __call__(self, losses, targets, weights, superpixels=None):
        if type(losses) == list:
            return losses
        return torch.zeros(self.loss_summary_length)


class MaskPokingLoss(nn.Module):
    def __init__(self, uncertainty):
        super(MaskPokingLoss, self).__init__()
        self.uncertainty = uncertainty
        self.loss = focal_loss_function(1)
        self.register_buffer('dummy_weight', torch.tensor(1, dtype=torch.float32))

    def forward(self, poking_scores, poking_targets):
        foreground, background = poking_targets[:, 0], poking_targets[:, 1]
        objectness, uncertainty = poking_scores[:, 0], poking_scores[:, 1]
        objectness_loss = self.loss(objectness, foreground, background, self.dummy_weight)
        if self.uncertainty:
            unc_foreground = (objectness >= 0) * background + (objectness <= 0) * foreground
            unc_background = (objectness > 0) * foreground + (objectness < 0) * background
        else:
            unc_foreground = foreground
            unc_background = background
        uncertainty_loss = self.loss(uncertainty, unc_foreground, unc_background, self.dummy_weight)
        return objectness_loss + uncertainty_loss
