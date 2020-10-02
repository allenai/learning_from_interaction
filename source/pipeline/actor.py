import numpy as np
import torch
import multiprocessing.dummy as mp
from json import load
from random import choice, randint, sample, shuffle
from copy import deepcopy
from skimage.segmentation import felzenszwalb
from skimage.color import rgb2hsv
from skimage.measure import label
from sklearn.decomposition import PCA
from pycocotools.mask import decode
from ai2thor.controller import Controller

from models.model import Model
from losses.losses import LossFunction
from replay_memory.replay_memory import Memory
from config import global_config, actor_config


class ActorPool:
    """
    The Trainer object uses a pool of actors to collect data. The start method spawns a collection of Actor processes,
    with which this object communicates via pipes. The child processes send images of THOR scenes, receive poking
    locations predicted by the model, and return feedback from poking. This workflow happens in the make_batch method,
    which is the main method of this class.
    """

    def __init__(self, model: Model, loss_function: LossFunction, memory: Memory, size: int, ground_truth: int):
        self.model = model
        self.loss_function = loss_function
        self.memory = memory

        self.actor_device = torch.cuda.current_device() if global_config.distributed else global_config.actor_gpu
        self.model_device = torch.cuda.current_device() if global_config.distributed else global_config.model_gpu

        self.size = size
        self.actors = []
        self.ground_truth = ground_truth

        self.scene_loader = self._make_loader(actor_config.data_files[0] if ground_truth < 4 else
                                              actor_config.data_files[global_config.val_scenes])

        self.scenes = self._make_scenes(ground_truth)

        if ground_truth in [1, 3]:
            self.poking_grid = [dict(point=(i, j)) for i in range(global_config.grid_size)
                                for j in range(global_config.grid_size)]
        else:
            self.poking_grid = None

    def start(self):
        for _ in range(self.size):
            pipe = mp.Pipe()
            p = mp.Process(target=Actor, args=(pipe[1], self.actor_device, (global_config, actor_config),
                                               self.ground_truth))
            p.start()
            self.actors.append((p, pipe[0]))

    def stop(self):
        for p, pipe in self.actors:
            pipe.send('stop')
            pipe.recv()
            p.join()
        self.actors = []

    def make_batch(self, batch_size, num_pokes, episode, episodes, threshold=None):
        data, predictions, feedback, targets = [], [], [], []
        successful_pokes = 0
        while len(data) < batch_size:
            ''' Setting new scenes '''
            new_data = self.get_image_batch()
            data += new_data

            if self.ground_truth == 5:
                continue

            ''' Interacting with scenes '''

            new_predictions = self.get_predictions(new_data, num_pokes, episode, episodes, threshold)

            if self.ground_truth in [1, 3]:
                new_predictions = self.refine_predictions_with_gt(new_data, new_predictions)

            predictions += new_predictions
            new_feedback = self.get_actor_feedback(new_data, new_predictions)
            feedback += new_feedback

            new_targets, new_successes = self.process_feedback(new_data, new_predictions, new_feedback)

            targets += new_targets
            successful_pokes += new_successes

        if self.ground_truth == 5:
            return [d['image'] for d in data], [d['metadata'] for d in data]

        ''' organize data '''

        images = [d['image'] for d in data]

        if global_config.superpixels:
            superpixels = [d['superpixels'] for d in data]
            batch = (images, targets, superpixels)
        else:
            batch = (images, targets)

        if self.ground_truth == 4:
            return batch, [d['metadata'] for d in data], predictions

        ''' computing some statistics '''

        stats = self.compute_stats(batch, predictions) + [successful_pokes]

        return batch, stats

    def get_image_batch(self):
        for _, pipe in self.actors:
            scene = next(self.scene_loader) if actor_config.use_dataset else choice(self.scenes)
            pipe.send(scene)
        data = []
        for _, pipe in self.actors:
            data.append(pipe.recv())
        return data

    def get_predictions(self, data, num_pokes, episode, episodes, threshold):
        predictions = []
        new_images_torch = self.to_torch([d['image'] for d in data]).cuda(self.model_device)
        if self.ground_truth == 4:
            actions, pred_masks, preds, scores = \
                self.model.compute_masks(new_images_torch, threshold)
            preds = list(zip(*preds))
            for a, m, p, s in zip(actions, pred_masks, preds, scores):
                predictions.append(dict(action=a, masks=m, predictions=p, scores=s))
        else:
            actions, preds = \
                self.model.compute_actions(new_images_torch, num_pokes, episode, episodes)
            preds = list(zip(*preds))
            for a, p in zip(actions, preds):
                predictions.append(dict(action=a, predictions=p))
        return predictions

    def get_actor_feedback(self, data, predictions):
        feedback = []
        actions = [p['action'] for p in predictions]
        if self.ground_truth == 3:
            feedback += [prediction['masks'] for prediction in predictions]
        if self.ground_truth == 2:
            for d, action in zip(data, actions):
                masks = self.get_gt_masks(d, action)
                feedback.append(masks)
        elif self.ground_truth in [0, 1, 4]:
            for actor, action in zip(self.actors, actions):
                actor[1].send(action)
            for actor in self.actors:
                feedback.append(actor[1].recv())
        return feedback

    def process_feedback(self, data, predictions, feedback):
        actions = [p['action'] for p in predictions]
        superpixels = [d['superpixels'] for d in data] if global_config.superpixels else None
        targets, successes = self.loss_function.process_feedback(actions, feedback, superpixels)
        return targets, successes

    def compute_stats(self, batch, predictions):
        if self.ground_truth in [1, 3]:
            return [0] * self.loss_function.loss_summary_length
        targets = batch[1]
        model_predictions = [p['predictions'] for p in predictions]

        model_preds_torch = tuple(torch.stack(list(p)) for p in zip(*model_predictions))
        targets_torch = tuple(torch.from_numpy(np.stack(t)).cuda(self.model_device) for t in zip(*targets))

        if global_config.superpixels:
            superpixels = batch[2]
            superpixels_torch = torch.from_numpy(np.stack(superpixels)).cuda(self.model_device)
        else:
            superpixels_torch = [None] * model_preds_torch[0].shape[0]

        weights = torch.ones(model_preds_torch[0].shape[0]).cuda(self.model_device)

        losses = self.loss_function(model_preds_torch, targets_torch, weights, superpixels_torch)

        if self.loss_function.prioritized_replay:
            losses = losses[0]

        return [loss.item() for loss in losses]

    # Below: Functionality for training with different levels of supervision.

    def refine_predictions_with_gt(self, data, predictions):
        assert actor_config.instance_only, 'Ground truth training not available for models with force prediction'
        for datapoint, prediction in zip(data, predictions):
            prediction['action'], prediction['masks'] = self.refine_actions_with_gt(prediction['action'], datapoint)
        return predictions

    def refine_actions_with_gt(self, actions, data):
        assert actor_config.instance_only, 'Ground truth training not available for models with force prediction'
        masks = data['metadata'][0][0]
        num_pokes = len(actions)
        reachable_pixels = data['image'][1] < actor_config.handDistance

        masks_reachable = [self.downsample(mask * reachable_pixels) for mask in masks]
        masks = [self.downsample(mask) for mask, maskr in zip(masks, masks_reachable) if np.any(maskr)]
        masks_reachable = [mask for mask in masks_reachable if np.any(mask)]

        if masks_reachable:
            bg = sum(masks_reachable) == 0
        else:
            bg = np.ones((global_config.grid_size, global_config.grid_size), dtype=np.bool)

        negatives = [action for action in actions if bg[action['point'][0], action['point'][1]]]
        negatives += sample(self.poking_grid, num_pokes - len(negatives))

        positives, selected_masks = [], []
        for mask, maskr in zip(masks, masks_reachable):
            hits = [ac for ac in actions if maskr[ac['point'][0], ac['point'][1]]][:actor_config.max_poke_keep]
            if hits:
                positives += hits
                selected_masks += [mask] * len(hits)
            else:
                poking_points = np.stack(np.where(maskr))
                k = min(poking_points.shape[1], actor_config.max_poke_attempts)
                random_points = [dict(point=tuple(poking_points[:, i]))
                                 for i in sample(range(poking_points.shape[1]), k)]
                positives += random_points
                selected_masks += [mask] * len(random_points)

        negatives = negatives[:num_pokes - len(positives)]
        actions = positives + negatives
        selected_masks += [np.zeros((global_config.grid_size, global_config.grid_size), dtype=np.bool)
                           for _ in range(len(negatives))]
        return actions[:num_pokes], selected_masks[:num_pokes]

    def get_gt_masks(self, data_point, action):
        assert actor_config.instance_only, 'Ground truth training not available for models with force prediction'
        half = (global_config.stride - 1) // 2
        masks = data_point['metadata'][0][0]
        depth = data_point['image'][1]
        if len(masks) == 0:
            return [np.zeros((global_config.grid_size, global_config.grid_size), dtype=np.bool)
                    for _ in range(len(action))]
        match_masks = np.stack(masks) * (depth < actor_config.handDistance)[None, ...]
        masks = [self.downsample(m) for m in masks]
        selected_gt_masks, already_selected = [], []
        for ac in action:
            x,y = ac['point']
            matches = list(np.where(match_masks[:, global_config.stride * x + half,
                                    global_config.stride * y + half])[0])
            if matches and not matches[0] in already_selected:
                selected_gt_masks.append(masks[matches[0]])
                if actor_config.remove_after_poke:
                    already_selected.append(matches[0])
            else:
                selected_gt_masks.append(np.zeros((global_config.grid_size, global_config.grid_size), dtype=np.bool))
        return selected_gt_masks

    def to_torch(self, images):
        images_torch = [self.memory.base_image_transform(image) for image in images]
        return torch.stack(images_torch)

    @staticmethod
    def downsample(mask):
        half = (global_config.stride - 1) // 2
        mask = mask[half:, half:]
        return mask[::global_config.stride, ::global_config.stride]

    @staticmethod
    def _make_loader(file):
        data = load(open(file, 'r'))
        scenes = [dict(scene=d['thor_meta']['scene'],
                       position=d['thor_meta']['position'],
                       rotation=d['thor_meta']['rotation'],
                       horizon=d['thor_meta']['horizon'],
                       seed=d['thor_meta']['seed']) for d in data]
        shuffle(scenes)
        return iter(scenes * 3)

    @staticmethod
    def _make_scenes(ground_truth):
        # Use if no fixed dataset of THOR positions is desired.
        train_scenes = ['FloorPlan%d_physics' % i for i in range(1, 29)] + \
                       ['FloorPlan%d_physics' % i for i in range(201, 229)] + \
                       ['FloorPlan%d_physics' % i for i in range(301, 329)]
        held_out = ['FloorPlan%d_physics' % i for i in range(29, 31)] + \
                   ['FloorPlan%d_physics' % i for i in range(229, 231)] + \
                   ['FloorPlan%d_physics' % i for i in range(329, 331)] + \
                   ['FloorPlan%d_physics' % i for i in range(401, 407)]
        train_scenes = [dict(scene=scene) for scene in train_scenes]
        held_out = [dict(scene=scene) for scene in held_out]
        scenes = [train_scenes, held_out]
        return scenes[0] if ground_truth < 4 else scenes[global_config.val_scenes]

    def make_batch_from_json(self, json_path):
        with open(json_path, 'r') as file:
            scenes = load(file)
        batched_scenes = [scenes[i * len(self.actors): (i + 1) * len(self.actors)]
                          for i in range(len(scenes) // len(self.actors) + 1)]
        images = []
        for batch in batched_scenes:
            for (_, pipe), scene in zip(self.actors, batch):
                scene_small = dict(scene=scene['thor_meta']['scene'],
                                   position=scene['thor_meta']['position'],
                                   rotation=scene['thor_meta']['rotation'],
                                   horizon=scene['thor_meta']['horizon'],
                                   seed=scene['thor_meta']['seed'])
                pipe.send(scene_small)
            for _, pipe in self.actors:
                images.append(pipe.recv()['image'])
        masks, gtms = [], []
        for scene in scenes:
            mask, gtm = [], []
            for annotation in scene['annotations']:
                mask.append(decode(annotation['segmentation']))
                gtm.append(self.round_force(annotation['min_force'], annotation['gt_mass']))
            masks.append(mask)

        return images, masks, gtms

    def round_force(self, force, mass):
        if force == -1:
            return self.round_mass(mass)
        if force < actor_config.force_buckets[1]:
            return 0
        if force < actor_config.force_buckets[2]:
            return 1
        return 2

    @staticmethod
    def round_mass(mass):
        if mass < actor_config.mass_buckets[0]:
            return 0
        if mass < actor_config.mass_buckets[1]:
            return 1
        return 2


class Actor:
    """
    The basic THOR actor class that we use. This process can be controlled via pipes. The main functionality is to
    provide RGBD views from THOR scenes, receive and perform pokes in these scenes, and provide the feedback.

    The feedback can be in the form of raw images, or already processed (for performance). The agent can run in standard
    mode (images are only extracted from iTHOR when the scene is at rest), or in video mode (images are extracted in
    real time; very slow).

    Additionally, this class can provide superpixel segmentations of the THOR scenes.
    """

    def __init__(self, pipe, gpu, configs, ground_truth=0, run=True):
        self.pipe = pipe
        self.gc, self.ac = configs
        self.ground_truth = ground_truth
        self.directions = [dict(x=a, y=b, z=c) for a in [-1, 1] for b in [1, 2] for c in [-1, 1]]
        self.controller = Controller(x_display='0.%d' % gpu, visibilityDistance=self.ac.visibilityDistance,
                                     renderDepthImage=self.gc.depth or ground_truth > 0,
                                     renderClassImage=ground_truth > 0, renderObjectImage=ground_truth > 0)
        self.grid = [(x, y) for x in range(self.gc.grid_size) for y in range(self.gc.grid_size)]

        self.depth_correction = self._make_depth_correction(self.gc.resolution, self.gc.resolution, 90)

        self.kernel_size = (self.ac.check_change_kernel.shape[0] - 1) // 2

        if run:
            self.run()

    @staticmethod
    def _make_depth_correction(height, width, vertical_field_of_view):
        focal_length_in_pixels = height / np.tan(vertical_field_of_view / 2 * np.pi / 180) / 2
        xs = (np.arange(height).astype(np.float32) - (height - 1) / 2) ** 2
        ys = (np.arange(width).astype(np.float32) - (width - 1) / 2) ** 2
        return np.sqrt(1 + (xs[:, None] + ys[None, :]) / focal_length_in_pixels ** 2)

    def run(self):
        while True:
            action = self.pipe.recv()
            if action == 'stop':
                break
            elif type(action) == dict:
                data = dict()
                image, superpixel, metadata = self.set_scene(action) if self.ac.use_dataset else self.set_scene0(action)
                data['image'] = image
                if self.gc.superpixels:
                    data['superpixels'] = superpixel
                if self.ground_truth:
                    data['metadata'] = metadata
                self.pipe.send(data)
            if type(action) == list:
                feedback = self.poke(action, superpixel)
                self.pipe.send(feedback)
        self.controller.stop()
        self.pipe.send('stop')

    def set_scene(self, scene_data):
        if self.ac.video_mode:
            self.controller.step(action='UnpausePhysicsAutoSim')
        scene, position, rotation, horizon, seed = (scene_data['scene'], scene_data['position'], scene_data['rotation'],
                                                    scene_data['horizon'], scene_data['seed'])
        self.controller.reset(scene)
        self.controller.step(action='InitialRandomSpawn', seed=seed,
                             forceVisible=True, numPlacementAttempts=5)
        self.controller.step(action='MakeAllObjectsMoveable')

        event = self.controller.step(action='TeleportFull', x=position['x'], y=position['y'],
                                     z=position['z'], rotation=rotation, horizon=horizon)

        image = event.frame

        if self.gc.superpixels:
            superpixel = felzenszwalb(image, scale=200, sigma=.5,
                                      min_size=200)[::self.gc.stride, ::self.gc.stride].astype(np.int32)
        else:
            superpixel = None

        if self.gc.depth:
            depth = event.depth_frame
            if self.gc.correct_depth:
                depth = (depth - .1) * self.depth_correction  # convert depth from camera plane to distance from camera
            image = (image, depth)

        if self.ground_truth:
            ground_truth = self.compute_ground_truth(event)
            metadata = (ground_truth, seed, position, rotation, horizon)
        else:
            metadata = None

        return image, superpixel, metadata

    def set_scene0(self, scene):
        if self.ac.video_mode:
            self.controller.step(action='UnpausePhysicsAutoSim')
        self.controller.reset(scene['scene'])
        seed = randint(0, 2 ** 30)
        self.controller.step(action='InitialRandomSpawn', seed=seed,
                             forceVisible=True, numPlacementAttempts=5)
        self.controller.step(action='MakeAllObjectsMoveable')
        event = self.controller.step(action='GetReachablePositions')
        positions = deepcopy(event.metadata['reachablePositions'])

        position, rotation, horizon = choice(positions), choice([0., 90., 180., 270.]), choice([-30., 0., 30., 60.])
        event = self.controller.step(action='TeleportFull', x=position['x'], y=position['y'],
                                     z=position['z'], rotation=rotation, horizon=horizon)

        if self.gc.respawn_until_object:
            contains_interactable_object = len([o for o in event.metadata['objects']
                                                if o['visible'] and (o['moveable'] or o['pickupable'])]) > 0
            while not contains_interactable_object:
                position, rotation, horizon = choice(positions), choice([0., 90., 180., 270.]), \
                                              choice([-30., 0., 30., 60.])
                event = self.controller.step(action='TeleportFull', x=position['x'], y=position['y'],
                                             z=position['z'], rotation=rotation, horizon=horizon)
                contains_interactable_object = len([o for o in event.metadata['objects']
                                                    if o['visible'] and (o['moveable'] or o['pickupable'])]) > 0

        image = event.frame

        if self.gc.superpixels:
            superpixel = felzenszwalb(image, scale=200, sigma=.5,
                                      min_size=200)[::self.gc.stride, ::self.gc.stride].astype(np.int32)
        else:
            superpixel = None

        if self.gc.depth:
            depth = event.depth_frame
            if self.gc.correct_depth:
                depth = (depth - .1) * self.depth_correction  # convert depth from camera plane to distance from camera
            image = (image, depth)

        if self.ground_truth:
            ground_truth = self.compute_ground_truth(event)
            metadata = (ground_truth, seed, position, rotation, horizon)
        else:
            metadata = None

        return image, superpixel, metadata

    def poke(self, action, superpixel):
        feedback = []

        if self.ac.video_mode:
            im1 = self.controller.step(action='PausePhysicsAutoSim').frame
        else:
            im1 = self.controller.step(action='Pass').frame
        for poke_point in action:
            if self.ac.instance_only:
                im2 = self.touch(poke_point['point'], self.ac.force_buckets[-1])
                poke_feedback = self.compute_feedback(im1, im2, poke_point['point'], superpixel)
            elif self.ac.scaleable:
                poke_feedback, im2 = self.touch_with_forces(im1, poke_point, superpixel)
            else:
                poke_feedback, im2 = self.touch_with_forces_nonscaleable(im1, poke_point, superpixel)
            feedback.append(poke_feedback)
            im1 = im2[-1]
        return feedback

    def touch_with_forces(self, im1, point_and_force, superpixel):
        direction = choice(self.directions)
        point, force = point_and_force['point'], point_and_force['force']
        smaller_force = max(force - 1, 0)
        im2 = self.touch(point, self.ac.force_buckets[smaller_force], direction)
        vis_feedback = self.compute_feedback(im1, im2, point, superpixel)
        if self.get_score(vis_feedback, point) > 1.5:
            return (vis_feedback, -1 if force > 0 else 0), im2
        im1 = im2[-1]
        im2 = self.touch(point, self.ac.force_buckets[force], direction)
        vis_feedback = self.compute_feedback(im1, im2, point, superpixel)
        if self.get_score(vis_feedback, point) > 1.5:
            return (vis_feedback, 0), im2
        if force < len(self.ac.force_buckets) - 1:
            im1 = im2[-1]
            im2 = self.touch(point, self.ac.force_buckets[-1], direction)
            vis_feedback = self.compute_feedback(im1, im2, point, superpixel)
            if self.get_score(vis_feedback, point) > 1.5:
                return (vis_feedback, 1), im2
        return (vis_feedback, 2), im2

    def touch_with_forces_nonscaleable(self, im1, point_and_force, superpixel):
        point = point_and_force['point']
        smaller_force = 0
        im2 = self.touch(point, self.ac.force_buckets[smaller_force])
        vis_feedback = self.compute_feedback(im1, im2, point, superpixel)
        if self.get_score(vis_feedback, point) > 1.5:
            return (vis_feedback, 0), im2
        im1 = im2[-1]
        im2 = self.touch(point, self.ac.force_buckets[1])
        vis_feedback = self.compute_feedback(im1, im2, point, superpixel)
        if self.get_score(vis_feedback, point) > 1.5:
            return (vis_feedback, 1), im2
        im1 = im2[-1]
        im2 = self.touch(point, self.ac.force_buckets[-1])
        vis_feedback = self.compute_feedback(im1, im2, point, superpixel)
        if self.get_score(vis_feedback, point) > 1.5:
            return (vis_feedback, 1), im2
        return (vis_feedback, 2), im2

    def touch(self, point, force, direction=None):
        y, x = point  # x axis (=first axis) in numpy is y-axis (=second axis) in images (THOR)
        if direction is None:
            direction = choice(self.directions)
        im2 = []
        if self.ac.video_mode:
            self.controller.step(
                dict(action='TouchThenApplyForce',
                     x=x / self.gc.grid_size + 1 / 2 / self.gc.grid_size,
                     y=y / self.gc.grid_size + 1 / 2 / self.gc.grid_size,
                     direction=direction,
                     handDistance=self.ac.handDistance,
                     moveMagnitude=force))
            im2.append(self.controller.step(action='AdvancePhysicsStep', timeStep=0.01).frame)
            for _ in range(25):
                im2.append(self.controller.step(action='AdvancePhysicsStep', timeStep=0.05).frame)

        else:
            im2.append(self.controller.step(
                dict(action='TouchThenApplyForce',
                     x=x / self.gc.grid_size + 1 / 2 / self.gc.grid_size,
                     y=y / self.gc.grid_size + 1 / 2 / self.gc.grid_size,
                     direction=direction,
                     handDistance=self.ac.handDistance,
                     moveMagnitude=force)).frame)
        return im2

    def compute_feedback(self, im1, im2, poke_point, superpixel):
        if self.ac.raw_feedback:
            return im1, im2

        if self.ac.hsv:
            im1, im2 = rgb2hsv(im1), [rgb2hsv(im) for im in im2]
        else:
            im1, im2 = im1.astype(np.float32), [im.astype(np.float32) for im in im2]

        if self.ac.video_mode:
            diff = self.pca_diff([im1] + im2) if self.ac.pca else self.compute_mean_of([im1] + im2)
        else:
            im2 = im2[0]
            diff = [im1 - im2]

        if self.ac.smooth_mask:
            mask = self.make_smooth_mask(diff, [im1] + im2)
        else:
            mask = self.make_mask(diff)

        if superpixel is not None and self.ac.superpixel_postprocessed_feedback:
            if self.ac.smooth_mask:
                raise ValueError
            mask = self.smooth_mask_over_superpixels(mask, superpixel)
        if self.ac.connectedness_postprocessed_feedback:
            if self.ac.smooth_mask:
                raise ValueError
            mask = self.connected_component(mask, poke_point)

        return mask

    def make_mask(self, diff):
        diff = diff[0]
        scores = (diff ** 2).reshape(self.gc.grid_size, self.gc.stride,
                                     self.gc.grid_size, self.gc.stride, 3).mean(axis=(1, 3, 4))

        mask = scores > self.ac.pixel_change_threshold
        return mask

    def smooth_mask_over_superpixels(self, mask, superpixels):
        smoothed_mask = np.zeros_like(mask)
        superpixels = [superpixels == i for i in np.unique(superpixels)]
        for superpixel in superpixels:
            if mask[superpixel].sum() / superpixel.sum() > self.ac.superpixel_postprocessing_threshold:
                smoothed_mask[superpixel] = True
        return smoothed_mask

    def connected_component(self, mask, poke_point):
        x, y = poke_point
        b = mask[x, y]
        mask[x, y] = True
        fat_mask = self.fatten(mask) if self.ac.fatten else mask
        labels = label(fat_mask) * mask
        i = labels[x, y]
        labels[x, y] *= b
        mask[x, y] = b
        return labels == i

    # Below: Functionality for computing videoPCA soft masks

    def make_smooth_mask(self, diff, video):
        with torch.no_grad():
            diff = torch.from_numpy(diff.transpose(0, 3, 1, 2)).float()
            diff = torch.sqrt((diff ** 2).sum(dim=1)).unsqueeze(1)
            diff = torch.nn.functional.conv2d(diff, self.ac.kernel, padding=(self.ac.kernel.shape[-1] - 1) // 2)
            if (diff[0] > self.ac.soft_mask_threshold).sum() == 0:
                return np.zeros(video[0].shape[:-1], dtype=np.float32)
            diff = self.bn_torch(diff)
            mask = diff.squeeze(1) > .5
            mask_np = mask.numpy()
            return self.color_histogram_soft_mask(mask_np, mask, video)

    def color_histogram_soft_mask(self, mask, mask_torch, video):
        if mask[0].sum() == 0 or (~mask[0]).sum() == 0:
            return mask[0].astype(np.float32)
        if not self.ac.hsv:
            video = [rgb2hsv(im) for im in video]
        video = self.combine_colors(video)
        fg_histogram = np.histogram(video[mask], bins=self.ac.num_bins, density=True)[0]
        bg_histogram = np.histogram(video[~mask], bins=self.ac.num_bins, density=True)[0] + 1e-5
        image = video[0].reshape(-1)
        soft_mask = fg_histogram[image] / (fg_histogram[image] + bg_histogram[image])
        soft_mask = soft_mask.astype(np.float32).reshape(video[0].shape)
        soft_mask = self.center_soft_masks(soft_mask, mask_torch[0])
        return self.bn_np(self.hysteresis_threshold(self.bn_np(soft_mask)))

    def combine_colors(self, video):
        video = np.stack(video)
        ret = np.zeros(video.shape[:-1], dtype=np.int)
        ret += ((video[..., 1] * np.cos(2 * np.pi * video[..., 0]) + 1) / 2
                * (self.ac.colres1 - 1)).astype(np.int)
        ret += ((video[..., 1] * np.sin(2 * np.pi * video[..., 0]) + 1) / 2
                * (self.ac.colres2 - 1)).astype(np.int) * self.ac.colres1
        ret += (video[..., 2] * (self.ac.colres3 - 1)).astype(np.int) * self.ac.colres1 * self.ac.colres2
        return ret

    def hysteresis_threshold(self, mask):
        thresholding = self.fatten(self.fatten(mask > self.ac.hyst_thresholds[0]))
        thresholding *= mask > self.ac.hyst_thresholds[1]
        return mask * thresholding

    @staticmethod
    def compute_mean_of(images):
        mean = sum(images) / len(images)
        return np.stack([im - mean for im in images])

    def pca_diff(self, video):
        video = np.stack(video)
        bs = video.shape[0]
        video = video.reshape((bs, self.gc.grid_size, self.gc.stride,
                               self.gc.grid_size, self.gc.stride, 3)).mean(axis=(2, 4))
        video_shape = video[0].shape
        video = video.reshape((bs, -1))
        pca = PCA(n_components=self.ac.num_pca_components)
        pca.fit(video)
        reconstruction = pca.inverse_transform(pca.transform(video))
        diff = video - reconstruction
        return diff.reshape(*((bs,) + video_shape))

    def center_soft_masks(self, soft_mask, mask):
        with torch.no_grad():
            mask = mask.float().unsqueeze(0).unsqueeze(1)
            mask = torch.nn.functional.conv2d(mask, self.ac.centering_kernel,
                                              padding=(self.ac.centering_kernel.shape[-1] - 1) // 2).squeeze() > 1
        return soft_mask * mask.numpy()

    @staticmethod
    def fatten(mask):
        fat_mask = mask.copy()
        fat_mask[:-1] = fat_mask[:-1] | mask[1:]
        fat_mask[1:] = fat_mask[1:] | mask[:-1]
        fat_mask[:, :-1] = fat_mask[:, :-1] | mask[:, 1:]
        fat_mask[:, 1:] = fat_mask[:, 1:] | mask[:, :-1]
        return fat_mask

    @staticmethod
    def bn_torch(array):
        bs = array.shape[0]
        unsqueeze = (bs,) + (1,) * (len(array.shape) - 1)
        diffmin = array.view(bs, -1).min(1)[0].view(*unsqueeze)
        diffmax = array.view(bs, -1).max(1)[0].view(*unsqueeze)
        return (array - diffmin) / (diffmax - diffmin + 1e-6)

    @staticmethod
    def bn_np(array):
        bs = array.shape[0]
        unsqueeze = (bs,) + (1,) * (len(array.shape) - 1)
        diffmin = array.reshape((bs, -1)).min(1).reshape(unsqueeze)
        diffmax = array.reshape((bs, -1)).max(1).reshape(unsqueeze)
        return (array - diffmin) / (diffmax - diffmin + 1e-6)

    def compute_ground_truth(self, event):
        depth = (event.depth_frame - 0.1) * self.depth_correction
        reachable_pixels = depth < self.ac.handDistance

        keys = [o['objectId'] for o in event.metadata['objects'] if
                o['objectId'] in event.instance_masks.keys() and
                o['visible'] and (o['moveable'] or o['pickupable']) and o['mass'] < self.ac.mass_threshold]
        masses_unf = [self.round_mass(o['mass'])
                      for o in event.metadata['objects'] if
                      o['objectId'] in event.instance_masks.keys() and
                      o['visible'] and (o['moveable'] or o['pickupable']) and o['mass'] < self.ac.mass_threshold]

        masks_unf = [event.instance_masks[key] for key in keys]
        masks, masses = [], []
        for mask, mass in zip(masks_unf, masses_unf):
            if self.ac.max_pixel_threshold > (mask * reachable_pixels).sum() > self.ac.min_pixel_threshold:
                masks.append(mask)
                masses.append(mass)

        poking_points = [np.stack(np.where(mask * reachable_pixels))
                         for mask in masks]
        poking_points = [[self.round(*tuple(points[:, i])) for i in sample(range(points.shape[1]),
                                                                           k=min(points.shape[1],
                                                                                 self.ac.max_poke_attempts))]
                         for points in poking_points]
        return masks, poking_points, masses

    def round(self, x, y):
        return x // self.gc.stride, y // self.gc.stride

    def round_mass(self, mass):
        if mass < self.ac.mass_buckets[0]:
            return 0
        if mass < self.ac.mass_buckets[1]:
            return 1
        return 2

    def get_score(self, mask, action):
        x, y = action
        dx1 = min(x, self.kernel_size)
        dx2 = min(self.gc.grid_size - 1 - x, self.kernel_size) + 1
        dy1 = min(y, self.kernel_size)
        dy2 = min(self.gc.grid_size - 1 - y, self.kernel_size) + 1
        x1, x2, y1, y2 = x - dx1, x + dx2, y - dy1, y + dy2
        return (mask[x1:x2, y1:y2] * self.ac.check_change_kernel[self.kernel_size - dx1:
                                                                 self.kernel_size + dx2,
                                                                 self.kernel_size - dy1:
                                                                 self.kernel_size + dy2]).sum()
