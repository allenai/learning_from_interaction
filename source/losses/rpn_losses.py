import torch
from torch.nn.functional import pad

from config import ObjectnessRPNLossConfig
from losses.losses import ObjectnessLoss
from config import global_config
from models.rpn_models import RoIModule


class ObjectnessRPNLoss(ObjectnessLoss):
    """
    This loss assumes that the model predicts: anchor box objectness logits, mask logits for a choice of anchor boxes
    (anchor box selection is greedy / not differentiable), pixel objectness logits, (optional filter logit)

    The losses for objectness and filter logit are described in the parent class.
        -   The loss for anchor box objectness is focal loss, with positive/negative targets if IoU of anchor box with
            poking mask is sufficiently large/small. I.e. standard loss for RPN
        -   The loss for mask logits is focal loss, with targets the poking mask that has highest IoU with anchor box
            corresponding to the mask logits. This is almost (but not quite) like in Mask R-CNN.
    """

    def __init__(self, loss_config: ObjectnessRPNLossConfig):
        super(ObjectnessRPNLoss, self).__init__(loss_config)
        self.loss_summary_length = 3 + loss_config.regression + global_config.superpixels + loss_config.filter
        self.roi_module = RoIModule(loss_config.roi_config)
        self.anchor_regression_loss = anchor_regression_loss_function(loss_config.regression_weight)

    def __call__(self, model_predictions: tuple, targets: tuple, weights, superpixels=None):
        poking_scores, masks, anchor_box_scores, anchor_box_regressions, _, selected_anchors = model_predictions
        device = masks.device
        object_masks, foreground_masks, background_masks = targets

        filter_and_smoothness_losses = \
            self.compute_filter_and_smoothness_loss(None, (poking_scores,), object_masks, weights, superpixels)

        anchor_score_losses, mask_losses, objectness_losses = [], [], []
        anchor_regression_losses = [] if self.config.regression else None

        for roi_masks, scores, regressions, anchors, objectness, masks, foreground, background, weight in zip(
                masks, anchor_box_scores, anchor_box_regressions, selected_anchors, poking_scores,
                object_masks, foreground_masks, background_masks, weights):

            poking_locations = foreground + background
            with torch.no_grad():
                masks_cum = pad(masks.cumsum(-1).cumsum(-2), [1, 0, 1, 0])
                poking_cum = pad(poking_locations.cumsum(-1).cumsum(-2), [1, 0, 1, 0])

            # RPN stage loss
            regressed_anchors = self.make_anchors_for_loss(regressions)
            anchor_score_targets, regression_matches = self.roi_module.match_anchors(regressed_anchors, regressions,
                                                                                     masks_cum, poking_cum)
            positives, negatives = anchor_score_targets

            anchor_score_loss = self.focal_loss(scores, positives.float(), negatives.float(), weight)
            anchor_score_losses.append(anchor_score_loss)

            if self.config.regression:
                anchor_regression_loss = self.compute_anchor_regression_loss(regression_matches, weight, device)
                anchor_regression_losses.append(anchor_regression_loss)

            # Mask stage loss
            mask_loss = self.compute_mask_loss(roi_masks, anchors, masks, masks_cum, weight)
            mask_losses.append(mask_loss)

            # Poking loss
            objectness_loss = self.compute_objectness_loss(objectness, foreground, background, weight)
            objectness_losses.append(objectness_loss)

        losses = [anchor_score_losses, mask_losses, objectness_losses]
        if anchor_regression_losses is not None:
            losses.append(anchor_regression_losses)

        losses = self.stack_losses(losses, device)

        losses += filter_and_smoothness_losses

        if self.prioritized_replay:
            priorities = self.compute_priorities([mask_losses])
            return losses, priorities

        return losses

    def compute_anchor_regression_loss(self, matches, weight, device):
        if matches is None:
            anchor_regression_loss = torch.tensor(0, dtype=torch.float32).to(device)
        else:
            anchor_regression_loss = self.anchor_regression_loss(*(matches + (weight,))) / matches[0].shape[0]

        return anchor_regression_loss

    def compute_mask_loss(self, roi_masks, selected_anchors, masks, masks_cum, weight):
        ious, _ = self.roi_module.compute_ious(masks_cum.unsqueeze(0), anchor_boxes=selected_anchors.unsqueeze(0))
        matched_masks = self.roi_module.match_masks(ious.squeeze(0), masks)
        if self.config.robustify is not None:
            matched_masks = self.robustify_targets(roi_masks, matched_masks)
        positives = matched_masks.float()
        negatives = 1 - positives
        return self.focal_loss(roi_masks, positives, negatives, weight)

    def robustify_targets(self, roi_masks, matched_masks, step=0.1, max_iter=25):
        # This is similar to using the "robust set loss" proposed in
        # Pathak et al. "Learning instance segmentation by interaction"
        iou_threshold = self.config.robustify
        with torch.no_grad():
            log_probs = roi_masks.clone()
            indexer = torch.zeros_like(matched_masks, dtype=torch.bool)

            for i in range(max_iter):
                iou_orig = self.iou(log_probs > 0, matched_masks)
                unconverged = ~(iou_orig > iou_threshold)
                if not torch.any(unconverged):
                    break

                indexer *= False
                indexer[unconverged] = matched_masks[unconverged]
                log_probs[indexer] += step
                iou_up_in = self.iou(log_probs > 0, matched_masks)

                log_probs[indexer] -= step
                indexer *= False
                indexer[unconverged] = ~ matched_masks[unconverged]
                log_probs[indexer] -= step
                iou_down_out = self.iou(log_probs > 0, matched_masks)

                indexer *= False
                indexer[unconverged] = matched_masks[unconverged]
                log_probs[indexer] += step

                improved_in = (iou_up_in > iou_orig) * unconverged
                indexer *= False
                indexer[improved_in] = ~ matched_masks[improved_in]
                log_probs[indexer] += step

                improved_out = (~ improved_in) * (iou_down_out > iou_orig) * unconverged
                indexer *= False
                indexer[improved_out] = matched_masks[improved_out]
                log_probs[indexer] -= step

            new_masks = log_probs > 0

            return new_masks

    @staticmethod
    def iou(mask1, mask2):
        intersection = (mask1 * mask2).sum(dim=(-2, -1)).float()
        union = (mask1 | mask2).sum(dim=(-2, -1)).float().clamp(min=1)
        return intersection / union

    def make_anchors_for_loss(self, regressions):
        anchors = []
        for delta in self.config.deltas:
            anchors.append(self.roi_module.make_regressed_anchors(regressions, delta))
        return torch.stack(anchors)

    def cuda(self, k: int):
        self.roi_module.cuda(k)
        return self


def anchor_regression_loss_function(hyperparameter):
    class AnchorRegressionLoss(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, boxes, masks, weight):
            space = tuple(masks.shape[-2:])
            masks = masks.unsqueeze(0).expand(9, -1, -1, -1)
            mask_areas = masks[..., -1, -1]
            box_areas = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
            anch0 = boxes[..., 0].reshape(-1)
            anch1 = boxes[..., 1].reshape(-1)
            anch2 = boxes[..., 2].reshape(-1)
            anch3 = boxes[..., 3].reshape(-1)
            flat_masks = masks.reshape(*((-1,) + space))
            m = torch.arange(anch0.shape[0])
            int0 = flat_masks[m, anch0, anch1]
            int1 = flat_masks[m, anch2, anch3]
            int2 = flat_masks[m, anch0, anch3]
            int3 = flat_masks[m, anch2, anch1]
            intersections = (int0 + int1 - int2 - int3).view(9, -1).float()
            ious = intersections / (mask_areas + box_areas - intersections).clamp(min=1)
            mask = ious[1:] < ious[0].unsqueeze(0)
            ctx.save_for_backward(logits, mask, weight)
            return mask.float().mean()

        @staticmethod
        def backward(ctx, dummy_grad):
            logits, mask, weight = ctx.saved_tensors
            mask = mask.view(4, 2, -1)
            sign = (mask[:, 0, :].float() - mask[:, 1, :].float())
            return - sign.t() * (-logits ** 2).exp() * hyperparameter * weight, None, None, None

    return AnchorRegressionLoss.apply
