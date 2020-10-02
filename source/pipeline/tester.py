import os
import json
import pickle
import numpy as np
import torch
from copy import deepcopy
from PIL.ImageDraw import Draw
from PIL.Image import fromarray
from pycocotools.mask import encode, area, toBbox
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

from config import TestingConfig, global_config
from models.model import Model
from replay_memory.replay_memory import Memory
from pipeline.actor import ActorPool, actor_config
from losses.losses import LossFunction


class Tester:
    """
    The main testing / illustration pipeline.
    """

    def __init__(self, model: Model, memory: Memory, loss_function: LossFunction, tester_config: TestingConfig):
        self.config = tester_config

        self.model_device = torch.cuda.current_device() if global_config.distributed else global_config.model_gpu

        self.model = model
        self.model.eval()

        self.loss_function = loss_function
        self.memory = memory

        self.actors = None

    def evaluate_coco_metrics(self, dataset: str or int, threshold: float, annotation_type='segm', iou=.5):  # or 'bbox'
        if type(dataset) == int:
            dataset = self.make_eval_dataset(dataset)
            path = './tmpdataset%d' % np.random.randint(0, 10 ** 10, 1)
            self.save_pycoco_compatible_json(path + 'no_mass.json', dataset, use_mass=0)
            self.save_pycoco_compatible_json(path + 'mass.json', dataset, use_mass=1)
            gt = True
        else:
            path = dataset
            dataset = self.load_dataset(path)
            gt = False

        model_predictions = self.predict_masks(dataset[0], threshold)
        inference_path = path + 'inf'
        self.save_predictions_to_json(inference_path + 'no_mass.json', model_predictions, use_mass=0)
        self.save_predictions_to_json(inference_path + 'mass.json', model_predictions, use_mass=1)

        confusion_matrix, corcoef = self.get_confusion_matrix(model_predictions, dataset)
        print(corcoef)
        print(confusion_matrix)
        confusion_matrix = confusion_matrix / confusion_matrix.sum()
        print(confusion_matrix)
        marginal = confusion_matrix.sum(axis=0)
        accuracy = sum(confusion_matrix[i, i] for i in range(3))
        class_wise_accuracies = [confusion_matrix[i, i] / confusion_matrix[:, i].sum() for i in range(3)]
        print(accuracy)
        print(marginal)
        print(class_wise_accuracies)
        print(sum(class_wise_accuracies) / 3)

        coco_gt_no_mass = COCO(path + 'no_mass.json')
        coco_gt_mass = COCO(path + 'mass.json')
        coco_inf_no_mass = COCO(inference_path + 'no_mass.json')
        coco_inf_mass = COCO(inference_path + 'mass.json')
        coco_eval_no_mass = COCOeval(coco_gt_no_mass, coco_inf_no_mass, annotation_type)
        coco_eval_mass = COCOeval(coco_gt_mass, coco_inf_mass, annotation_type)
        coco_eval_no_mass.params.iouThrs = np.array([iou] + list(coco_eval_no_mass.params.iouThrs)[1:])
        coco_eval_mass.params.iouThrs = np.array([iou] + list(coco_eval_mass.params.iouThrs)[1:])
        coco_eval_no_mass.evaluate()
        coco_eval_mass.evaluate()
        coco_eval_no_mass.accumulate()
        coco_eval_mass.accumulate()
        pr_curve_no_mass = coco_eval_no_mass.eval['precision'][0, :, 0, 0, -1]
        pr_curve_mass = coco_eval_mass.eval['precision'][0, :, 0, 0, -1]
        coco_eval_no_mass.summarize()
        coco_eval_mass.summarize()

        os.remove(inference_path + 'no_mass.json')
        os.remove(inference_path + 'mass.json')
        if gt:
            os.remove(path + 'no_mass.json')
            os.remove(path + 'mass.json')
        print(pr_curve_no_mass)
        print(pr_curve_mass)
        return pr_curve_no_mass

    def make_eval_dataset(self, dataset_size):
        self.actors = ActorPool(self.model, self.loss_function, self.memory, self.config.num_actors, 5)
        self.actors.start()
        images, metadata = self.actors.make_batch(dataset_size, None, None, None)
        self.actors.stop()
        masks, metadata = [m[0][0] for m in metadata], [(m[0][1:],) + m[1:] for m in metadata]
        return images, masks, metadata

    def predict_masks(self, images: list, threshold: float):
        batches = [images[i:i + self.config.bs] for i in range(0, len(images), self.config.bs)]
        masks, scores, masses = [], [], []
        for batch in batches:
            batch_torch = torch.stack([self.memory.base_image_transform(image)
                                       for image in batch]).cuda(self.model_device)
            actions, new_masks, _, new_scores = self.model.compute_masks(batch_torch, threshold)
            masks += new_masks
            scores += new_scores
            masses += [[a['force'] for a in action] for action in actions]
        return masks, scores, masses

    def save_predictions_to_json(self, path: str, predictions: tuple, use_mass=0):
        data = []
        images = []
        predicted_masses = [0, 0, 0]
        k = 0
        for i, (masks, scores, masses) in enumerate(zip(*predictions)):
            d = dict(width=global_config.resolution, height=global_config.resolution, id=i)
            images.append(d)
            for mask, score, mass in zip(masks, scores, masses):
                assert mass in [0, 1, 2], 'masses wrong in predictions'
                predicted_masses[mass] += 1
                k += 1
                segmentation, bbox, ar = self.compute_annotations(self.upsample(mask))
                d = dict(image_id=i, category_id=int(use_mass * mass + 1), score=score,
                         segmentation=segmentation, bbox=bbox, area=ar, id=k)
                data.append(d)

        file = open(path, 'w')
        json.dump(dict(annotations=data, images=images), file)
        file.close()
        print(predicted_masses)

    def save_pycoco_compatible_json(self, path: str, dataset: tuple, use_mass=0):
        coco_dataset = dict()
        coco_dataset['info'] = {'dataset_size': len(dataset),
                                'depth': global_config.depth,
                                'mass_threshold': actor_config.mass_threshold,
                                'min_pixel_threshold': actor_config.min_pixel_threshold,
                                'max_pixel_threshold': actor_config.max_pixel_threshold}
        coco_dataset['licenses'] = {}
        if use_mass:
            coco_dataset['categories'] = [{'supercategory': 'object',
                                           'id': 1,
                                           'name': 'light'},
                                          {'supercategory': 'object',
                                           'id': 2,
                                           'name': 'medium'},
                                          {'supercategory': 'object',
                                           'id': 3,
                                           'name': 'heavy'}
                                          ]
        else:
            coco_dataset['categories'] = [{'supercategory': 'object',
                                           'id': 1,
                                           'name': 'object'}]
        coco_dataset['images'] = []
        coco_dataset['annotations'] = []
        k = 0
        instance_areas = []
        for i, (_, masks, metadata) in enumerate(zip(*dataset)):
            coco_dataset['images'].append(
                {'width': global_config.resolution, 'height': global_config.resolution, 'id': i})
            masses = metadata[0][1]
            for instance_mask, mass in zip(masks, masses):
                assert mass in [0, 1, 2], 'masses from metadata wrong'
                k += 1
                segmentation, bbox, ar = self.compute_annotations(instance_mask)
                instance_areas.append(ar)
                coco_dataset['annotations'].append(
                    {'iscrowd': 0, 'image_id': i, 'id': k, 'category_id': use_mass * mass + 1,
                     'area': ar, 'segmentation': segmentation, 'bbox': bbox})

        file = open(path, 'w')
        json.dump(coco_dataset, file)
        file.close()
        print('NUMBER OF INSTANCES IN DATASET: %d' % k)
        mean_area = sum(instance_areas) / max(len(instance_areas), 1)
        print('MEAN AREA OF INSTANCE IN DATASET: %f' % mean_area)

    def make_and_save_dataset(self, dataset_size, path):
        dataset = self.make_eval_dataset(dataset_size)
        self.save_data(dataset, path + '.pickle')
        self.save_pycoco_compatible_json(path + 'no_mass.json', dataset, use_mass=0)
        self.save_pycoco_compatible_json(path + 'mass.json', dataset, use_mass=1)

    @staticmethod
    def save_data(dataset, path):
        file = open(path, 'wb')
        pickle.dump(dataset, file)
        file.close()

    @staticmethod
    def load_dataset(path):
        return pickle.load(open(path + '.pickle', 'rb'))

    @staticmethod
    def compute_annotations(instance, encoding='ascii', transpose=True):
        if transpose:
            segmentation = encode(instance.astype(np.uint8).T)
        else:
            segmentation = encode(np.asarray(instance.astype(np.uint8), order="F"))
        segmentation['counts'] = segmentation['counts'].decode(encoding)
        bbox = list(toBbox(segmentation))
        ar = int(area(segmentation))
        return segmentation, bbox, ar

    @staticmethod
    def upsample(mask):
        return mask.repeat(global_config.stride, axis=0).repeat(global_config.stride, axis=1)

    def illustrate_poking_and_predictions(self, path: str, num_images: int, threshold: float):
        """
        :param path: The path to the folder where illustrations are to be saved
        :param num_images: Number of images to illustrate
        :param threshold: The value of the hyperparameter that controls the number of object proposals of the model.
        :return: Nothing. It fills the folder with images

        Attention: The code clears the current contents of the folder!

        This only works for models that provide pixel-wise predictions.

        The illustrations include:
            a) Image with GT overlay
            b) Image with predicted masks and masses overlay. Masses are red=light, green=medium, blue=heavy
            c) interaction score heat map
            d) mass logits map (same color coding as above)
            e) self supervised ground truth
            f) raw image
            g) depth image

        """
        if path[-1] != '/':
            path = path + '/'
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file[-1] == 'g':
                    os.remove(path + file)
        else:
            os.mkdir(path)
        rgb = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        self.actors = ActorPool(self.model, self.loss_function, self.memory, self.config.num_actors, 4)
        self.actors.start()
        batch, metadatas, predictions = self.actors.make_batch(num_images, None, None, None, threshold=threshold)
        self.actors.stop()

        images_and_depths, targets = batch[:2]
        images = [image[0] if global_config.depth else image for image in images_and_depths]
        depths = [image[1] if global_config.depth else None for image in images_and_depths]
        targets = [target[0] for target in targets]
        gt_masks = [metadata[0][0] for metadata in metadatas]

        i = 0
        for image, depth, target, p, gtm in zip(images, depths, targets, predictions, gt_masks):
            objectness, masses, pred_mask, action = p['predictions'][0], p['predictions'][1], p['masks'], p['action']
            if depth is not None:
                depth = ((1 - depth / actor_config.handDistance).clip(min=0, max=1) * 255).astype(np.uint8)
                depth = fromarray(depth)
            image = fromarray(image)
            raw_image = deepcopy(image)
            image = image.convert('LA').convert('RGB')
            im_heatmap = deepcopy(image)
            im_massmap = deepcopy(image)
            im_poking_mask = deepcopy(image)
            im_gt_mask = deepcopy(image)
            draw = Draw(image)
            draw_heatmap = Draw(im_heatmap)
            draw_massmap = Draw(im_massmap)
            draw_poking_mask = Draw(im_poking_mask)

            poking_masks = [t for t in target if t.sum() > 0]

            masses = masses.argmax(dim=0).detach().cpu().numpy()
            objectness = objectness[0].sigmoid()

            for mask, color in zip(gtm, self.config.colors):
                mask = fromarray(mask.astype(np.uint8) * 170)
                im_gt_mask.paste(color, (0, 0), mask)

            for x in range(global_config.grid_size):
                for y in range(global_config.grid_size):
                    corners = self.corners(x, y)

                    draw_heatmap.rectangle(corners[0], outline=(0, int(255 * objectness[x, y]), 0))
                    draw_massmap.rectangle(corners[0], outline=rgb[masses[x, y]])

                    for color, poking_mask in zip(self.config.colors, poking_masks):
                        if poking_mask[x, y]:
                            draw_poking_mask.rectangle(corners[0], outline=color)

                    for color, m in zip(self.config.colors, pred_mask):
                        if m[x, y]:
                            draw.rectangle(corners[0], outline=color)

            for ac in action:
                point, force = ac['point'], ac['force']
                corners = self.corners(*point)
                draw_poking_mask.rectangle(corners[1], outline=(0, 255, 0), fill=(0, 255, 0))
                draw.rectangle(corners[1], outline=rgb[force], fill=rgb[force])

            i += 1
            raw_image.save(path + '%d_f.png' % i)
            im_gt_mask.save(path + '%d_a.png' % i)
            image.save(path + '%d_b.png' % i)
            im_heatmap.save(path + '%d_c.png' % i)
            im_massmap.save(path + '%d_d.png' % i)
            im_poking_mask.save(path + '%d_e.png' % i)
            if depth is not None:
                depth.save(path + '%d_g.png' % i)
            del draw
            del draw_heatmap
            del draw_massmap
            del draw_poking_mask

    @staticmethod
    def corners(x, y):
        x, y = y, x
        return [x * global_config.stride, y * global_config.stride,
                (x + 1) * global_config.stride, (y + 1) * global_config.stride], \
               [x * global_config.stride - global_config.stride // 3,
                y * global_config.stride - global_config.stride // 3,
                (x + 1) * global_config.stride + global_config.stride // 3,
                (y + 1) * global_config.stride + global_config.stride // 3]

    def get_confusion_matrix(self, predictions, dataset):
        cm = np.zeros((3, 3))
        preds, matched_gts = [], []
        gtms = [md[0][1] for md in dataset[2]]
        for pred_masks, pred_masses, gt_masks, gt_masses in zip(predictions[0], predictions[2], dataset[1], gtms):
            if len(pred_masks) == 0 or len(gt_masks) == 0:
                continue
            gt_masks = np.stack(gt_masks)
            assert gt_masks.shape[0] == len(gt_masses), 'different num gt_masses vs gt_masks'
            assert len(pred_masks) == len(pred_masses), 'different num pred_masses vs pred_masks'
            for mask, mass in zip(pred_masks, pred_masses):
                mask = self.upsample(mask)[None, ...]
                intersections = (mask * gt_masks).sum(axis=(1, 2))
                unions = (mask | gt_masks).sum(axis=(1, 2)).clip(min=1)
                ious = intersections.astype(np.float32) / unions
                best_match = ious.argmax()
                if ious[best_match] >= 0.5:
                    cm[mass, gt_masses[best_match]] += 1
                    preds.append(mass)
                    matched_gts.append(gt_masses[best_match])
        return cm, np.corrcoef(np.array([preds, matched_gts]))
