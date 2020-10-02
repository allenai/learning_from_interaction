import torch
from torch import nn
import numpy as np
from random import sample

from config import ClusteringModelConfig, global_config
from models.backbones import UNetBackbone  # , R50FPNBackbone
from models.model import Model


class ClusteringModel(Model):
    """
    This model consists of a backbone that produces features at resolution grid_size x grid_size, and three heads.

    The first head predicts a D x grid_size x grid_size tensor of D dimensional grid cell feature vectors.
    Segmentation masks are computed by clustering these feature vectors around the feature vectors of seed grid cells.

    The second head predicts confidence scores for the seed grid cells. Seed grid cells are greedily picked according
    to their confidence scores as poking locations, and to determine masks.

    The third head predicts three mass logits. These are sampled at poking locations, and determine the force used
    for poking.

    Optionally, the network also outputs a single scalar for each image, predicting whether the image contains any
    pokeable objects. This could be used in combination with an appropriate loss function that only backpropagates
    gradients into the feature vectors / confidence scores for those images that actually contain objects.
    """

    def __init__(self, model_config: ClusteringModelConfig):
        super(ClusteringModel, self).__init__()
        self.config = model_config
        if model_config.backbone == 'r50fpn':
            self.backbone = R50FPNBackbone()
        else:
            self.backbone = UNetBackbone(model_config)

        backbone_out_dim = 256 if model_config.backbone == 'r50fpn' else 64
        self.head = nn.Sequential(nn.Conv2d(backbone_out_dim + 2, 128, kernel_size=1), nn.ReLU(),
                                  nn.Conv2d(128, model_config.out_dim + 1 + self.config.uncertainty, 1))

        self.mass_head = nn.Sequential(nn.Conv2d(backbone_out_dim, 128, kernel_size=1), nn.ReLU(),
                                       nn.Conv2d(128, 3, 1))

        if model_config.filter:
            self.filter_head = nn.Sequential(nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0, bias=False),
                                             nn.BatchNorm2d(1024),
                                             nn.ReLU(),
                                             nn.Conv2d(1024, 1, kernel_size=1, stride=1, padding=0))

        self.poking_grid = [dict(point=(i, j), force=force) for i in range(global_config.grid_size)
                            for j in range(global_config.grid_size) for force in [0, 1, 2]]
        coordinate_embedding_grid = np.array([[[x, y] for y in range(global_config.grid_size)]
                                              for x in range(global_config.grid_size)]).transpose((2, 0, 1))
        self.register_buffer('coordinate_embedding_grid', torch.from_numpy(coordinate_embedding_grid).float())

        if model_config.distance_function == 'L2':
            self.distance_function = lambda x, y: ((x - y) ** 2).sum(0)
        elif model_config.distance_function == 'Cosine':
            distance_function = lambda x, y: 1 - (x * y).sum(0) / (np.linalg.norm(x, axis=0)
                                                                   * np.linalg.norm(y, axis=0) + 1e-4)
            inv = lambda x: x
            self.distance_function = lambda x, y: inv(distance_function(x, y))

    def forward(self, x, targets=None):
        """
        :param x: Shape BS x 3(+1) x resolution x resolution
        :param targets: Not used. This model is a single shot detection model.
        :return: (Shape BS x grid_size x grid_size, BS x 3 x grid_size x grid_size, BS x D x grid_size x grid_size),
                    BS (optional)
        """

        z, x3 = self.backbone(x)
        y = self.head(torch.cat([z, self.coordinate_embedding_grid.repeat(z.shape[0], 1, 1, 1)], dim=1))

        obj = y[:, self.config.out_dim:]
        mass = self.mass_head(z)

        ret = (obj, mass, y[:, :self.config.out_dim])

        if self.config.filter:
            ret = ret + (self.filter_head(x3).squeeze(),)

        return ret

    def compute_actions(self, images, pokes, episode, episodes):
        obj_pokes, random_pokes = pokes // 2, pokes - pokes // 2  # We chose half of actions random for exploration
        actions = []

        with torch.no_grad():
            out = self.forward(images)
            obj, m, emb = out[:3]
            embeddings = emb.cpu().numpy()
            objectnesss = obj[:, self.config.uncertainty].cpu().numpy()
            masss = m.argmax(dim=1).cpu().numpy()

        for embedding, objectness, mass in zip(embeddings, objectnesss, masss):
            action, i = [], 0
            while i < obj_pokes and objectness.max() > -10000:
                i += 1
                a, _ = self.action_and_mask(embedding, objectness, -10000)
                action.append(dict(point=a, force=mass[a[0], a[1]]))
            action += sample(self.poking_grid, random_pokes)
            actions.append(action)
        return actions, out

    def compute_masks(self, images, threshold):
        masks, scores, actions = [], [], []
        self.eval()
        with torch.no_grad():
            out = self.forward(images)
            obj, m, emb = out[:3]
            embeddings = emb.cpu().numpy()
            objectnesss = obj[:, 0].cpu().numpy()
            masss = m.argmax(dim=1).cpu().numpy()

        for embedding, objectness, mass in zip(embeddings, objectnesss, masss):
            action, mask, new_scores = [], [], []
            score = objectness.max()
            i = 0
            while score > threshold and i < global_config.max_pokes:
                new_scores.append(float(1 / (1 + np.exp(-score))))
                a, m = self.action_and_mask(embedding, objectness, threshold)
                mask.append(m)
                action.append(dict(point=a, force=mass[a[0], a[1]]))
                score = objectness.max()
                i += 1
            masks.append(mask)
            scores.append(new_scores)
            actions.append(action)
        return actions, masks, out, scores

    def action_and_mask(self, embedding, objectness, threshold):
        argmax = objectness.argmax()
        action = (argmax // global_config.grid_size, argmax % global_config.grid_size)
        mask = self.make_mask(embedding, objectness, action, threshold)
        return action, mask

    def make_mask(self, embedding, objectness, action, threshold):
        margin_threshold = self.config.margin_threshold[threshold > -10000]
        center = embedding[:, action[0], action[1]][:, None, None]
        distances = self.distance_function(embedding, center)
        mask = distances < self.config.threshold
        center = embedding[:, mask].mean(axis=1)[:, None, None]
        distances = self.distance_function(embedding, center)
        mask = distances < self.config.threshold
        if self.config.threshold != margin_threshold:
            mask2 = distances < margin_threshold
        else:
            mask2 = mask
        objectness[mask2] = threshold - 1
        if not self.config.overlapping_objects:
            embedding[:, mask2] = self.config.reset_value * np.ones_like(embedding[:, mask2])
        return mask

    @staticmethod
    def upsample(mask):
        return mask.repeat(global_config.stride, axis=0).repeat(global_config.stride, axis=1)

    def load(self, path):
        state_dict = torch.load(path, map_location='cuda:%d' % global_config.model_gpu)
        print(self.load_state_dict(state_dict, strict=False))
        if self.config.freeze:
            self.freeze_detection_net(False)

    def toggle_detection_net(self, freeze):
        for param in self.backbone.parameters():
            param.requires_grad = freeze
        for param in self.head.parameters():
            param.requires_grad = freeze

    def toggle_mass_head(self, freeze):
        for param in self.mass_head.parameters():
            param.requires_grad = freeze


class ClusteringModelPooled(Model):
    """
    Similar to ClusteringModel, but mass predictions are instance-wise, computed on the mean pooled features of the
    instance mask. In some sense, this is a two stage detection approach.
    """

    def __init__(self, model_config: ClusteringModelConfig):
        super(ClusteringModelPooled, self).__init__()
        self.config = model_config
        self.backbone = UNetBackbone(model_config)

        backbone_out_dim = 64
        self.head = nn.Sequential(nn.Conv2d(backbone_out_dim + 2, 128, kernel_size=1), nn.ReLU(),
                                  nn.Conv2d(128, model_config.out_dim + 1 + self.config.uncertainty, 1))

        self.mass_head = nn.Sequential(nn.Linear(64, 256, bias=False), nn.ReLU(),
                                       nn.Linear(256, 3))

        self.poking_grid = [((i, j), mass) for i in range(global_config.grid_size)
                            for j in range(global_config.grid_size) for mass in [0, 1, 2]]
        coordinate_embedding_grid = np.array([[[x, y] for y in range(global_config.grid_size)]
                                              for x in range(global_config.grid_size)]).transpose((2, 0, 1))
        self.register_buffer('coordinate_embedding_grid', torch.from_numpy(coordinate_embedding_grid).float())

        if model_config.distance_function == 'L2':
            self.distance_function = lambda x, y: ((x - y) ** 2).sum(0)
        elif model_config.distance_function == 'Cosine':
            distance_function = lambda x, y: 1 - (x * y).sum(0) / (np.linalg.norm(x, axis=0)
                                                                   * np.linalg.norm(y, axis=0) + 1e-4)
            inv = lambda x: x
            self.distance_function = lambda x, y: inv(distance_function(x, y))

    def forward(self, x, targets=None):
        z, x3 = self.backbone(x)
        y = self.head(torch.cat([z, self.coordinate_embedding_grid.repeat(z.shape[0], 1, 1, 1)], dim=1))

        obj = y[:, self.config.out_dim:]

        mass_logits = self.compute_mass_logits(z, targets)

        return obj, y[:, :self.config.out_dim], mass_logits, z

    def compute_actions(self, images, pokes, episode, episodes):
        obj_pokes, random_pokes = pokes // 2, pokes - pokes // 2
        actions = []

        with torch.no_grad():
            out = self.forward(images)
            obj, emb, _, z = out
            embeddings = emb.cpu().numpy()
            objectnesss = obj[:, self.config.uncertainty].cpu().numpy()

            for embedding, objectness, features in zip(embeddings, objectnesss, z):
                action, i = [], 0
                while i < obj_pokes and objectness.max() > -10000:
                    i += 1
                    a, m = self.action_and_mask(embedding, objectness, -10000)
                    force_logits = self.compute_single_action_mass_logits(m, features)
                    action.append(dict(point=a, force=force_logits.argmax().item()))
                action += sample(self.poking_grid, random_pokes)
                actions.append(action)
        return actions, out

    def compute_masks(self, images, threshold):
        masks, scores, actions = [], [], []
        self.eval()
        with torch.no_grad():
            out = self.forward(images)
            obj, emb, _, z = out
            embeddings = emb.cpu().numpy()
            objectnesss = obj[:, self.config.uncertainty].cpu().numpy()

            for embedding, objectness, features in zip(embeddings, objectnesss, z):
                action, mask, new_scores = [], [], []
                score = objectness.max()
                i = 0
                while score > threshold and i < global_config.max_pokes:
                    new_scores.append(float(1 / (1 + np.exp(-score))))
                    a, m = self.action_and_mask(embedding, objectness, threshold)
                    force_logits = self.compute_single_action_mass_logits(m, features)
                    mask.append(m)
                    action.append(dict(point=a, force=force_logits.argmax().item()))
                    score = objectness.max()
                    i += 1
                masks.append(mask)
                scores.append(new_scores)
                actions.append(action)
        return actions, masks, out, scores

    def action_and_mask(self, embedding, objectness, threshold):
        argmax = objectness.argmax()
        action = (argmax // global_config.grid_size, argmax % global_config.grid_size)
        mask = self.make_mask(embedding, objectness, action, threshold)
        return action, mask

    def make_mask(self, embedding, objectness, action, threshold):
        margin_threshold = self.config.margin_threshold[threshold > -10000]
        center = embedding[:, action[0], action[1]][:, None, None]
        distances = self.distance_function(embedding, center)
        mask = distances < self.config.threshold
        center = embedding[:, mask].mean(axis=1)[:, None, None]
        distances = self.distance_function(embedding, center)
        mask = distances < self.config.threshold
        if self.config.threshold != margin_threshold:
            mask2 = distances < margin_threshold
        else:
            mask2 = mask
        objectness[mask2] = threshold - 1
        if not self.config.overlapping_objects:
            embedding[:, mask2] = self.config.reset_value * np.ones_like(embedding[:, mask2])
        return mask

    def compute_mass_logits(self, z, targets):
        targets = targets[0] if targets is not None else [None] * z.shape[0]
        mass_logits = []
        for features, masks in zip(z, targets):
            logits = []
            if masks is None:
                logits.append(torch.zeros(3, device=z.device))
            else:
                for mask in masks:
                    logits.append(self.compute_single_action_mass_logits(mask, features))
            mass_logits.append(torch.stack(logits))
        return torch.stack(mass_logits)

    def compute_single_action_mass_logits(self, mask, features):
        if mask.sum() > 0:
            pooled_features = features[:, mask].mean(dim=1).unsqueeze(0)
            logits = self.mass_head(pooled_features).squeeze(0)
        else:
            logits = torch.zeros(3, dtype=torch.float32).to(features.device)
        return logits

    @staticmethod
    def upsample(mask):
        return mask.repeat(global_config.stride, axis=0).repeat(global_config.stride, axis=1)

    def load(self, path):
        state_dict = torch.load(path, map_location='cuda:%d' % global_config.model_gpu)
        print(self.load_state_dict(state_dict, strict=False))
        if self.config.freeze:
            self.toggle_detection_net(False)

    def toggle_detection_net(self, freeze):
        for param in self.backbone.parameters():
            param.requires_grad = freeze
        for param in self.head.parameters():
            param.requires_grad = freeze

    def toggle_mass_head(self, freeze):
        for param in self.mass_head.parameters():
            param.requires_grad = freeze
