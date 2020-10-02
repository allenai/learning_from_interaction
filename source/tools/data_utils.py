import json
import os

from torch.utils.data import Dataset

from config import global_config
from pipeline.actor import Actor


class EvalDataset(Dataset):
    def __init__(self, dataset_file, memory, controller):
        with open(dataset_file, "r") as f:
            self.dataset = json.load(f)
        self.folder = os.path.dirname(dataset_file)
        self.memory = memory
        self.lut = Actor._make_depth_correction(global_config.resolution, global_config.resolution, 90)
        self.controller = controller

    def __len__(self):
        return len(self.dataset)

    def load_meta(self, thor_meta):
        scene = thor_meta["scene"]
        seed = thor_meta["seed"]
        position = thor_meta["position"]
        rotation = thor_meta["rotation"]
        horizon = thor_meta["horizon"]

        self.controller.reset(scene)
        self.controller.step(action='InitialRandomSpawn', seed=seed,
                             forceVisible=True, numPlacementAttempts=5)
        self.controller.step(action='MakeAllObjectsMoveable')
        event = self.controller.step(action='TeleportFull', x=position['x'], y=position['y'],
                                     z=position['z'], rotation=rotation, horizon=horizon)

        return event

    def __getitem__(self, item):
        entry = self.dataset[item]
        evt = self.load_meta(entry["thor_meta"])
        rgb = evt.frame.copy()
        if global_config.depth:
            dist = (evt.depth_frame.copy() - .1) * self.lut
            rgbd = self.memory.base_image_transform((rgb, dist))
        else:
            rgbd = self.memory.base_image_transform(rgb)

        return rgbd, entry["image_id"]


class ActiveDataset(EvalDataset):
    def __init__(self, dataset_file, memory, controller, conn):
        super().__init__(dataset_file, memory, controller)
        self.conn = conn

    def process(self):
        while True:
            item = self.conn.recv()
            if item is None:
                break
            self.conn.send(self.__getitem__(item))
