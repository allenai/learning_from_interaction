import torch
from torch import nn
from random import sample

from config import ROIModuleConfig, global_config, RPNModelConfig
from models.backbones import UNetBackbone
from models.model import Model


class RPNWithMask(Model):
    """
    This is a DeepMask inspired region proposal network with instance mask and objectness score predictions. It can
    be trained self-supervised. Anchor box regression is optional.
    """
    def __init__(self, model_config: RPNModelConfig):
        super(RPNWithMask, self).__init__()
        self.config = model_config

        self.backbone = UNetBackbone(model_config)

        self.anchor_box_scoring_head = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, stride=2),
                                                     nn.BatchNorm2d(128),
                                                     nn.ReLU(),
                                                     nn.Conv2d(128, (1 + 4 * model_config.regression)
                                                               * model_config.num_anchors, kernel_size=2))

        self.masking_head = nn.Sequential(nn.Conv2d(64, 32, kernel_size=5, padding=2, bias=False),
                                          nn.BatchNorm2d(32),
                                          nn.ReLU(),
                                          nn.Conv2d(32, 32, kernel_size=5, padding=2, bias=False),
                                          nn.BatchNorm2d(32),
                                          nn.ReLU(),
                                          nn.Conv2d(32, 1, kernel_size=5, padding=2))

        self.poking_head = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(),
                                         nn.Conv2d(64, 1 + self.config.uncertainty, kernel_size=1))

        self.roi_module = RoIModule(model_config.roi_config)

        self.poking_grid = [(i, j) for i in range(global_config.grid_size) for j in range(global_config.grid_size)]

    def forward(self, images: torch.tensor, targets=None):

        y, x3 = self.backbone(images)

        scores_and_regression = self.anchor_box_scoring_head(x3)
        shape = tuple(scores_and_regression.shape)
        shape = (shape[0], -1, 1 + 4 * self.config.regression) + shape[2:]
        scores_and_regression = scores_and_regression.view(*shape)
        shape = tuple(scores_and_regression.shape)
        anchor_box_scores = scores_and_regression[:, :, 0]
        if self.config.regression:
            anchor_box_regression = scores_and_regression[:, :, 1:].transpose(2, 3).transpose(3, 4).contiguous()
        else:
            anchor_box_regression = torch.zeros(shape[:2] + shape[3:] + (4,), dtype=torch.float32).to(y.device)
        poking_scores = self.poking_head(y)

        anchors = self.roi_module.make_and_regress_anchors(anchor_box_regression)

        selected_anchors, selected_scores = self.roi_module.select_anchors(anchor_box_scores, anchors)

        masked_features = self.mask_features(y, selected_anchors)

        masks = self.masking_head(masked_features).reshape(y.shape[0], -1,
                                                           global_config.grid_size, global_config.grid_size)

        return poking_scores, masks, anchor_box_scores, anchor_box_regression, selected_scores, selected_anchors

    def compute_actions(self, images: torch.tensor, num_pokes: int, episode: int, episodes: int):
        obj_pokes, random_pokes = num_pokes // 2, num_pokes - num_pokes // 2
        actions = []

        with torch.no_grad():
            out = self.forward(images)
            poking_scores, _, _, _, _, selected_anchors = out
            poking_scores = poking_scores[:, self.config.uncertainty]
            for anchors, scores in zip(selected_anchors, poking_scores):
                action = []
                scores = scores.sigmoid()
                for i in range(obj_pokes):
                    argmax = (scores * self.make_anchor_mask(anchors[i])).argmax().item()
                    action.append(dict(point=(argmax // global_config.grid_size, argmax % global_config.grid_size)))
                action += sample(self.poking_grid, random_pokes)
                actions.append(action)
        return actions, out

    def compute_masks(self, images: torch.tensor, threshold: float):
        pred_masks, pred_scores, actions = [], [], []
        self.eval()
        with torch.no_grad():
            out = self.forward(images)
            poking_scores, masks, _, _, selected_scores, selected_anchors = out
            poking_scores = poking_scores[:, 0]

            for mask, scores, anchors, poking_score in zip(masks, selected_scores, selected_anchors, poking_scores):
                new_masks, new_scores, new_actions = [], [], []
                poking_score = poking_score.sigmoid()
                for m, s, a in zip(mask, scores, anchors):
                    if s < threshold:
                        break
                    new_masks.append((m > 0).cpu().numpy())
                    new_scores.append(s.sigmoid().item())
                    argmax = (self.make_anchor_mask(a) * poking_score).argmax().item()
                    new_actions.append(dict(point=(argmax // global_config.grid_size,
                                                   argmax % global_config.grid_size)))
                pred_masks.append(new_masks)
                pred_scores.append(new_scores)
                actions.append(new_actions)
        return actions, pred_masks, out, pred_scores

    def mask_features(self, features, anchors):
        anchor_mask = self.make_anchor_mask(anchors)
        return (features.unsqueeze(1) * anchor_mask.unsqueeze(2)).view(-1, features.shape[1],
                                                                       global_config.grid_size,
                                                                       global_config.grid_size)

    @staticmethod
    def make_anchor_mask(anchors):
        shape = tuple(anchors.shape[:-1]) + (global_config.grid_size, global_config.grid_size)
        anchors = anchors.view(-1, 4)
        anchor_mask = torch.zeros((anchors.shape[0], global_config.grid_size, global_config.grid_size),
                                  dtype=torch.bool).to(anchors.device)
        for i in range(anchors.shape[0]):
            anchor_mask[i, anchors[i, 0]:anchors[i, 2], anchors[i, 1]:anchors[i, 3]] = True

        return anchor_mask.view(*shape)


class RoIModule(nn.Module):
    def __init__(self, model_config: ROIModuleConfig):
        super(RoIModule, self).__init__()
        self.config = model_config
        self.clamp = lambda x: min(max(x, 0), global_config.grid_size - 1)
        self.coarse_grid_stride = global_config.grid_size // self.config.coarse_grid_size

        self._init_anchors()

    def _init_anchors(self):
        boxes = torch.from_numpy(self.config.boxes).int()
        self.register_buffer('anchor_offsets_and_sizes', boxes.unsqueeze(1).unsqueeze(2))
        anchor_sizes = boxes[:, 2:].repeat(1, 2).contiguous().unsqueeze(1).unsqueeze(2).float()
        self.register_buffer('anchor_sizes', anchor_sizes)
        positive_thresholds = torch.tensor(self.config.positive_thresholds)
        negative_thresholds = torch.tensor(self.config.negative_thresholds)
        self.register_buffer('positive_thresholds',
                             positive_thresholds.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4))
        self.register_buffer('negative_thresholds',
                             negative_thresholds.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4))
        coarse_grid_coordinates = torch.tensor([[[x * self.coarse_grid_stride, y * self.coarse_grid_stride]
                                                 for y in range(self.config.coarse_grid_size)]
                                                for x in range(self.config.coarse_grid_size)]).int()
        self.register_buffer('coarse_grid_coordinates', coarse_grid_coordinates.contiguous().unsqueeze(0))
        anchors = self.make_regressed_anchors(torch.zeros(self.config.num_anchors, self.config.coarse_grid_size,
                                                          self.config.coarse_grid_size, 4))
        self.register_buffer('anchor_boxes', anchors)

    def make_regressed_anchors(self, regression_logits, delta=(0, 0, 0, 0)):
        with torch.no_grad():
            regression = (regression_logits.sigmoid() * self.anchor_sizes / 3).int() + self.anchor_offsets_and_sizes
            delta = torch.tensor(list(delta)).unsqueeze(0).unsqueeze(1).unsqueeze(2).to(regression.device)
            anchors = regression + delta
            anchors[:, :, :, :2] = anchors[:, :, :, :2] + self.coarse_grid_coordinates
            anchors[:, :, :, 2:] = anchors[:, :, :, :2] + anchors[:, :, :, 2:]
            return anchors.clamp(min=0, max=global_config.grid_size)

    def select_anchors(self, anchor_box_scores: torch.tensor, anchors=None):
        stride1 = self.config.coarse_grid_size ** 2
        stride2 = self.config.coarse_grid_size
        if anchors is None:
            anchors = [None] * anchor_box_scores.shape[0]
        selected_anchors = []
        selected_scores = []
        for scores, anchs in zip(anchor_box_scores, anchors):
            scores_numpy = scores.detach().cpu().numpy()
            indices = []
            for _ in range(self.config.num_rois):
                ind = scores_numpy.argmax()
                ind = (ind // stride1, (ind % stride1) // stride2, ind % stride2)
                if anchs is not None:
                    with torch.no_grad():
                        mask = self.box_iou_mask(anchs[ind[0], ind[1], ind[2]], anchs)
                        scores_numpy[mask] = scores_numpy[mask] - 5
                else:
                    scores_numpy[ind[0], ind[1], ind[2]] = - 1000
                indices.append(ind)

            selected_anchors.append(torch.stack([anchs[i[0], i[1], i[2]] for i in reversed(indices)]))
            selected_scores.append(torch.stack([scores[i[0], i[1], i[2]] for i in reversed(indices)]))

        return torch.stack(selected_anchors), torch.stack(selected_scores)

    def box_iou_mask(self, box, boxes):
        with torch.no_grad():
            area_box = (box[2] - box[0]) * (box[3] - box[1])
            area_boxes = (boxes[:, :, :, 2] - boxes[:, :, :, 0]) * (boxes[:, :, :, 3] - boxes[:, :, :, 1])
            min_xy = torch.max(box[:2], boxes[:, :, :, :2])
            max_xy = torch.min(box[2:], boxes[:, :, :, 2:])
            diff = (max_xy - min_xy).clamp(min=0)
            intersections = (diff[:, :, :, 0] * diff[:, :, :, 1]).float()
            ious = intersections / (area_box + area_boxes - intersections).clamp(min=1)
            mask = (ious > self.config.nms_threshold).cpu().numpy()
            return mask

    def refine_indices(self, masks, indices, poking_locations):
        ious, intersections = self.compute_anchor_targets(masks, poking_locations)
        with torch.no_grad():
            positive_anchors = ((ious / self.positive_thresholds) > 1).sum(dim=1) > 0
            positive_anchors = positive_anchors * (intersections > self.config.poking_filter_threshold)
            positive_indices = torch.nonzero(positive_anchors)
        refined_indices = [index for index in indices if index in positive_indices][-self.config.num_rois:]
        if len(refined_indices) < self.config.num_rois:
            refined_indices = [index for index in indices if index not in
                               positive_indices][-(self.config.num_rois - len(refined_indices)):] + \
                              refined_indices
        return refined_indices

    def compute_ious(self, masks, poking_locations=None, anchor_boxes=None):
        if anchor_boxes is None:
            anchor_boxes = self.anchor_boxes.unsqueeze(0)
        with torch.no_grad():
            size = tuple(masks.shape[:2]) + tuple(anchor_boxes.shape[1:-1])
            space = tuple(masks.shape[-2:])
            for _ in range(len(anchor_boxes.shape[1:-1])):
                masks = masks.unsqueeze(2)
            masks = masks.expand(*(size + space))
            anchor_boxes_unsqueeze = anchor_boxes.unsqueeze(1).expand(*(size + (4,)))
            mask_areas = masks[..., -1, -1]
            box_areas = (anchor_boxes_unsqueeze[..., 2] - anchor_boxes_unsqueeze[..., 0]) * \
                        (anchor_boxes_unsqueeze[..., 3] - anchor_boxes_unsqueeze[..., 1])
            anch0 = anchor_boxes_unsqueeze[..., 0].reshape(-1)
            anch1 = anchor_boxes_unsqueeze[..., 1].reshape(-1)
            anch2 = anchor_boxes_unsqueeze[..., 2].reshape(-1)
            anch3 = anchor_boxes_unsqueeze[..., 3].reshape(-1)
            flat_masks = masks.reshape(*((-1,) + space))
            m = torch.arange(anch0.shape[0])
            int0 = flat_masks[m, anch0, anch1]
            int1 = flat_masks[m, anch2, anch3]
            int2 = flat_masks[m, anch0, anch3]
            int3 = flat_masks[m, anch2, anch1]
            intersections = (int0 + int1 - int2 - int3).view(*size).float()
            ious = intersections / (mask_areas + box_areas - intersections).clamp(min=1)

            intersections = None

            if poking_locations is not None:
                size = tuple(anchor_boxes.shape[:-1])
                poking_locations = poking_locations.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(*(size + space))
                anch0 = anchor_boxes[..., 0].reshape(-1)
                anch1 = anchor_boxes[..., 1].reshape(-1)
                anch2 = anchor_boxes[..., 2].reshape(-1)
                anch3 = anchor_boxes[..., 3].reshape(-1)
                flat_poke = poking_locations.reshape(*((-1,) + space))
                m = torch.arange(anch0.shape[0])
                int0 = flat_poke[m, anch0, anch1]
                int1 = flat_poke[m, anch2, anch3]
                int2 = flat_poke[m, anch0, anch3]
                int3 = flat_poke[m, anch2, anch1]
                intersections = (int0 + int1 - int2 - int3).view(*size).float()

            return ious, intersections

    def make_and_regress_anchors(self, regressions):
        anchors = []
        for regression_logits in regressions:
            anchors.append(self.make_regressed_anchors(regression_logits))
        return torch.stack(anchors)

    def match_anchors(self, regressed_anchors, regressions, masks, poking_locations):

        ious, intersections = self.compute_ious(masks.unsqueeze(0),
                                                poking_locations.unsqueeze(0),
                                                regressed_anchors[0].unsqueeze(0))
        ious, intersections = ious.squeeze(0), intersections.squeeze(0)

        with torch.no_grad():
            positives = ((ious / self.positive_thresholds) > 1).sum(dim=1) > 0
            negatives = ((ious / self.negative_thresholds) > 1).sum(dim=1) == 0
            positives = positives * (intersections > self.config.poking_filter_threshold)
            negatives = negatives * (intersections > self.config.poking_filter_threshold)
            positives, negatives = positives.squeeze(0), negatives.squeeze(0)
            anchor_targets = (positives, negatives)

        matches = None

        if anchor_targets[0].sum() > 0:
            positive_regressions = regressions[positives]
            ious = ious[:, positives]
            matched_masks = self.match_masks(ious, masks)
            positive_boxes = regressed_anchors[:, positives]
            matches = (positive_regressions, positive_boxes, matched_masks)

        return anchor_targets, matches

    @staticmethod
    def match_masks(ious, masks):
        mask_shape = len(masks.shape[:-2])
        resolution = masks.shape[-1]
        anchor_shape = tuple(ious.shape[mask_shape:])
        ious = ious.view(*((-1,) + anchor_shape))
        masks = masks.view(-1, resolution, resolution)
        matches = ious.argmax(0).view(-1)
        return masks[matches].view(*(anchor_shape + (resolution, resolution)))
