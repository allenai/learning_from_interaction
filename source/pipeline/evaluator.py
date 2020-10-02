import time
from datetime import datetime
import json
from collections import OrderedDict
import os
from multiprocessing import Process, Pipe

from ai2thor.controller import Controller
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import numpy as np
import torch

from pipeline.tester import Tester
from config import global_config, actor_config
from tools.logger import LOGGER, init_logging
from tools.data_utils import ActiveDataset


class Evaluator(Tester):
    @staticmethod
    def save_predictions_to_json(path: str, predictions: tuple, image_ids: list = None,
                                 interactable_classes: int or list = 0):
        data = []
        images = []
        k = 0
        save_time = time.time()
        last_eta = save_time
        nitems = len(predictions[0])

        for i, preds in enumerate(zip(*predictions)):
            imid = i if image_ids is None else image_ids[i]
            d = dict(width=global_config.resolution, height=global_config.resolution, id=imid)
            images.append(d)
            for mask, score, action in zip(*preds):
                if isinstance(interactable_classes, list):
                    current_category = interactable_classes[action["force"].item()]
                else:
                    current_category = interactable_classes
                k += 1
                segmentation, bbox, ar = Evaluator.compute_annotations(
                    Evaluator.upsample(mask), encoding='utf-8', transpose=False
                )
                d = dict(image_id=imid, category_id=current_category, score=score,
                         segmentation=segmentation, bbox=bbox)
                data.append(d)
            new_time = time.time()
            if new_time - last_eta >= 10:
                curtime = new_time - save_time
                eta = curtime / (i + 1) * (nitems - i - 1)
                LOGGER.info("save {}/{} spent {} s ETA {} s".format(i + 1, nitems, curtime, eta))
                last_eta = new_time

        with open(path, 'w') as file:
            json.dump(dict(annotations=data, images=images), file)

    @staticmethod
    def run_load(dataset_path, memory, conn):
        init_logging()
        controller = Controller(
            x_display='0.%d' % global_config.actor_gpu,
            visibilityDistance=actor_config.visibilityDistance,
            renderDepthImage=global_config.depth
        )
        dataset = ActiveDataset(dataset_path, memory, controller, conn)
        dataset.process()
        controller.stop()

    def push_pull_items(self, begin_next, loaders, ndata):
        begin_item, next_item = begin_next
        last_item = next_item - 1

        # Preload data
        for last_item in range(next_item, min(next_item + self.config.bs, ndata)):
            loaders[last_item % len(loaders)][1].send(last_item)

        # Read former data
        ims, names = [], []
        if begin_item >= 0:
            for read_item in range(begin_item, min(begin_item + self.config.bs, ndata)):
                im, name = loaders[read_item % len(loaders)][1].recv()
                ims.append(im)
                names.append(name)

        # Update pointers
        begin_item, next_item = next_item, last_item + 1

        return (begin_item, next_item), ims, names

    def dataset_forward(self, dataset_path: str, interaction_thres: float):
        loaders = []
        try:
            for c in range(self.config.num_actors):
                parent, child = Pipe()
                proc = Process(target=Evaluator.run_load, args=(dataset_path, self.memory, child))
                proc.start()
                loaders.append((proc, parent))
            dataset = ActiveDataset(dataset_path, self.memory, None, None)
            ndata = len(dataset)

            begin_next, _, _ = self.push_pull_items((-1, 0), loaders, ndata)

            image_ids = []
            masks, scores, actions = [], [], []
            eval_time = time.time()
            last_eta = eval_time
            while begin_next[0] < ndata:
                begin_next, ims, names = self.push_pull_items(begin_next, loaders, ndata)

                batch_torch = torch.stack(ims).cuda(self.model_device)
                image_ids.extend(names)

                new_actions, new_masks, _, new_scores = self.model.compute_masks(batch_torch, interaction_thres)

                masks += new_masks
                scores += new_scores
                actions += new_actions

                new_time = time.time()
                if new_time - last_eta >= 10:
                    curtime = new_time - eval_time
                    nepisodes = begin_next[0]
                    eta = curtime / nepisodes * (ndata - nepisodes)
                    LOGGER.info("forward {}/{} spent {} s ETA {} s".format(nepisodes, ndata, curtime, eta))
                    last_eta = new_time
        finally:
            for it, (loader, conn) in enumerate(loaders):
                conn.send(None)
            for it, (loader, conn) in enumerate(loaders):
                LOGGER.info("Joining loader {}".format(it))
                loader.join()

        return (masks, scores, actions), image_ids

    @staticmethod
    def coco_eval(coco_gt: COCO, coco_dt: COCO, annotation_type: str, use_categories: bool = True):
        coco_eval = COCOeval(coco_gt, coco_dt, annotation_type)

        if not use_categories:
            coco_eval.params.useCats = 0

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
        results = OrderedDict(
            (metric,  float(coco_eval.stats[idx] if coco_eval.stats[idx] >= 0 else "nan"))
            for idx, metric in enumerate(metrics)
        )

        return results

    def inference(self, dataset_path: str, output_folder: str, interaction_thres: float, id: int or str = 0,
                  interactable_classes: int or list = 0):
        date_time = datetime.now().strftime("%Y%m%d%H%M%S")
        data_basename = os.path.basename(dataset_path)
        output_path = os.path.join(output_folder, data_basename.replace(
            ".json", "__{}__thres{}__{}__inf.json".format(id, interaction_thres, date_time)
        ))
        LOGGER.info("Using inference results path {}".format(output_path))
        model_predictions, image_ids = self.dataset_forward(dataset_path, interaction_thres)
        self.save_predictions_to_json(output_path, model_predictions, image_ids, interactable_classes)

        return output_path

    @staticmethod
    def load_coco_detections(coco_gt, coco_dets_path: str, annotation_type='segm'):
        with open(coco_dets_path, "r") as f:
            raw_inf = json.load(f)

        if not isinstance(raw_inf, list):
            assert raw_inf["images"][0]["id"] in coco_gt.imgs,\
                "ensure {} was generated with Evaluator's inference".format(coco_dets_path)
            raw_inf = raw_inf["annotations"]
            for entry in raw_inf:
                entry.pop("id", None)
                entry.pop("area", None)

        if annotation_type == 'segm':  # pop bounding boxes to avoid using bbox areas
            for entry in raw_inf:
                entry.pop("bbox", None)

        # LOGGER.info("{} annotations in {}".format(len(raw_inf), coco_dets_path))
        return coco_gt.loadRes(raw_inf)

    @staticmethod
    def results_string(res, labs=None):
        if labs is None:
            labs = res.keys()
        return ", ".join(["%s %4.2f%%" % (m, res[m] * 100) for m in labs])

    @staticmethod
    def evaluate(coco_gt_path: str, coco_dets_path: str, annotation_type='bbox', labs=("AP50", "AP"),
                 use_categories=False):
        if annotation_type == 'mass':
            return Evaluator.evaluate_mass(coco_gt_path, coco_dets_path)

        coco_gt = COCO(coco_gt_path)
        coco_dt = Evaluator.load_coco_detections(coco_gt, coco_dets_path, annotation_type)

        res = Evaluator.coco_eval(coco_gt, coco_dt, annotation_type, use_categories=use_categories)
        res_str = Evaluator.results_string(res, labs)
        LOGGER.info("RESULTS {} {} {}".format(
            coco_dets_path,
            annotation_type,
            res_str
        ))
        return {annotation_type: res}

    @staticmethod
    def confusion_matrix(coco_gt, coco_dt, nclasses=3):
        eval = COCOeval(coco_gt, coco_dt, "bbox")
        eval.params.useCats = 0
        eval.params.iouThrs = [0.5]
        eval.params.areaRng = eval.params.areaRng[:1]
        eval.params.areaRngLbl = eval.params.areaRngLbl[:1]
        eval.evaluate()

        conf_mat = np.zeros((nclasses, nclasses), dtype=np.int64)
        for it, im in enumerate(eval.evalImgs):
            if im is not None:
                for gt, dt in zip(im['gtIds'], list(im['gtMatches'][0].astype(np.int32))):
                    if dt > 0:
                        row = eval.cocoDt.anns[dt]["category_id"]
                        col = eval.cocoGt.anns[gt]["category_id"]
                        conf_mat[row, col] += 1

        return conf_mat

    @staticmethod
    def evaluate_mass(coco_gt_path: str, coco_dets_path: str):
        annotation_type = 'bbox'
        coco_gt = COCO(coco_gt_path)
        coco_dt = Evaluator.load_coco_detections(coco_gt, coco_dets_path, annotation_type)

        res = Evaluator.coco_eval(coco_gt, coco_dt, annotation_type, use_categories=True)

        conf_mat = Evaluator.confusion_matrix(coco_gt, coco_dt)
        accuracies = conf_mat.diagonal() / conf_mat.sum(axis=0)
        mean_accuracy = accuracies.mean().item()

        res.update({"accuracy": mean_accuracy})

        res_str = Evaluator.results_string(res, labs=("AP50", "accuracy"))
        LOGGER.info("RESULTS {} {} {}".format(
            coco_dets_path,
            "mass",
            res_str
        ))

        return {"mass": res}
