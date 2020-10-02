import numpy as np
import torch


class GlobalConfig:
    def __init__(self):
        self.resolution = 300
        self.grid_size = 100
        self.stride = self.resolution // self.grid_size
        self.depth = True
        self.respawn_until_object = False
        self.superpixels = True
        self.use_of = False
        self.model_gpu = 0
        self.actor_gpu = 1
        self.of_gpu = 7
        self.max_pokes = 32  # This limit is required to support one hot encoding of poking masks
        self.val_scenes = 1
        self.distributed = False
        self.correct_depth = True


global_config = GlobalConfig()


class ActorConfig:
    def __init__(self):
        # configs for self-supervision module
        self.video_mode = global_config.use_of
        self.raw_feedback = False
        self.superpixel_postprocessed_feedback = True
        self.superpixel_postprocessing_threshold = .25
        self.connectedness_postprocessed_feedback = False
        self.fatten = False
        self.pixel_change_threshold = .01
        self.hsv = True
        self.check_change_kernel = np.array([[np.exp(-np.sqrt((x - 2) ** 2 + (y - 2) ** 2) / 1)
                                              for x in range(5)] for y in range(5)])

        # configs for videoPCA
        self.pca = False and self.video_mode
        self.num_pca_components = 4
        self.smooth_mask = False and self.pca
        kernel = torch.tensor([[np.exp(-np.sqrt((x - 7) ** 2 + (y - 7) ** 2) / 2 / 5 ** 2)
                                for x in range(15)]
                               for y in range(15)]).unsqueeze(0).unsqueeze(1)
        self.pca_smoothing_kernel = kernel / kernel.sum()
        centering_kernel = torch.tensor([[np.exp(-np.sqrt((x - 20) ** 2 + (y - 20) ** 2) / 2 / 5 ** 2)
                                          for x in range(41)]
                                         for y in range(41)]).unsqueeze(0).unsqueeze(1)
        self.pca_centering_kernel = centering_kernel
        self.soft_mask_threshold = .005
        self.colres1 = 100  # 100
        self.colres2 = 25  # 25
        self.colres3 = 10  # 10
        self.num_color_bins = np.arange(self.colres1 * self.colres2 * self.colres3) + 1
        self.hyst_thresholds = (.8, .5)

        # configs for interaction
        self.instance_only = False
        self.force = 250
        self.force_buckets = [5, 30, 200]
        self.scaleable = True
        self.handDistance = 1.5
        self.visibilityDistance = 10
        self.max_poke_attempts = 3
        self.max_poke_keep = 1
        self.remove_after_poke = False

        # The following attributes filter the objects counted in the ground truth
        self.mass_buckets = [.5, 2.]
        self.mass_threshold = 150
        self.max_pixel_threshold = 300 ** 2
        self.min_pixel_threshold = 10
        self.data_files = ['unary_dataset__detectron2__60_30_30__train.json',
                           'unary_dataset__detectron2__60_30_30__valid.json']
        self.use_dataset = True


actor_config = ActorConfig()


class BackboneConfig:
    def __init__(self):
        self.small = False


class ModelConfigFgBg(BackboneConfig):
    def __init__(self):
        super(ModelConfigFgBg, self).__init__()
        self.uncertainty = False
        self.superpixel = False
        self.fatten = True


class ClusteringModelConfig(BackboneConfig):
    def __init__(self):
        super(ClusteringModelConfig, self).__init__()
        self.backbone = 'unet'  # unet or r50fpn
        self.out_dim = 16
        self.max_masks = global_config.max_pokes
        self.overlapping_objects = False
        self.filter = False
        self.uncertainty = int(False)
        self.distance_function = 'L2'  # L2 or Cosine
        self.threshold = 1  # 1./.9 for L2/Cosine
        self.margin_threshold = (1, 1)
        self.reset_value = 10000  # 10000 / 0 for L2 / Cosine
        self.use_coordinate_embeddings = True
        self.freeze = False


class ROIModuleConfig:
    def __init__(self):
        self.boxes = np.array([[0, 0, 9, 9],
                               [0, 10, 9, 9],
                               [10, 0, 9, 9],
                               [10, 10, 9, 9],
                               [4, 4, 9, 9],
                               [-4, -4, 9, 9],
                               [-4, 4, 9, 9],
                               [4, -4, 9, 9],
                               [0, 0, 19, 19],
                               [0, 10, 19, 19],
                               [10, 0, 19, 19],
                               [10, 10, 19, 19],
                               [0, 4, 19, 9],
                               [4, 0, 9, 19],
                               [-9, -4, 19, 9],
                               [-4, -9, 9, 19],
                               [0, 0, 39, 39]])  # offset_x, offset_y, delta_x, delta_y
        '''
        base stride is 60 pixels = 20 grid cells
        5 30x30 boxes / cell inside the base grid
        3 30x30 boxes / cell half way between neighbouring grid cells
        4 60x60 boxes / cell (1 centered, 3 half way) 
        2 30x60 box
        2 60x30 box
        1 120x120 box

        IoU thresholds for small cells should be tighter than for large ones

        ulc = upper left corners are at 0, 20, 40, 60, 80
        views will be [ulc + offset, ulc + offset + delta + 1]
        '''
        self.positive_thresholds = [.35] * 8 + [.25] * 4 + [.3] * 4 + [.2]
        self.negative_thresholds = [.2] * 8 + [.15] * 4 + [.2] * 4 + [.1]
        self.num_anchors = len(self.boxes)
        self.num_rois = 16
        self.coarse_grid_size = 5
        self.poking_filter_threshold = 2.5
        self.nms_threshold = .4


class RPNModelConfig(BackboneConfig):
    def __init__(self):
        super(RPNModelConfig, self).__init__()
        self.roi_config = ROIModuleConfig()
        self.teacher_forcing = False
        self.num_anchors = self.roi_config.num_anchors
        self.nms = True
        self.regression = False
        self.uncertainty = int(False)


class MemoryConfigPIL:
    def __init__(self):
        self.capacity = 20000
        self.prioritized_replay = True
        self.bias_correct = False
        self.warm_start_memory = None
        self.flip_prob = .5
        self.jitter_prob = .8
        self.jitter = .3
        self.initial_priority = .5


class MemoryConfigTensor:
    def __init__(self):
        self.capacity = 20000
        self.warm_start_memory = None
        self.num_workers = 0
        self.sizes = [(3 + global_config.depth, global_config.resolution, global_config.resolution),
                      (global_config.max_pokes, global_config.grid_size, global_config.grid_size),
                      (global_config.grid_size, global_config.grid_size),
                      (global_config.grid_size, global_config.grid_size)] + \
                     ([(global_config.grid_size, global_config.grid_size)] if global_config.superpixels else [])
        self.dtypes = [torch.float32,  # image
                       torch.bool,  # obj_masks
                       torch.float32,  # foreground
                       torch.float32] + ([torch.int32] if global_config.superpixels else [])  # background


class ObjectnessLossConfig:
    def __init__(self):
        self.filter = False  # has to match the corresponding attribute in model config file
        self.filter_threshold = -.3
        self.prioritized_replay = True
        self.foreground_threshold = 1.5
        self.objectness_weight = 1
        self.smoothness_weight = 0
        self.kernel = actor_config.check_change_kernel
        self.kernel_size = (self.kernel.shape[0] - 1) // 2
        self.check_change_kernel = actor_config.check_change_kernel
        self.superpixel_for_action_feedback = False
        self.robustify = None
        self.point_feedback_for_action = False
        self.localize_object_around_poking_point = True
        self.prioritize_default = .5
        self.prioritize_function = lambda score: (score - .5) ** 2 + .02


class ObjectnessClusteringLossConfig(ObjectnessLossConfig):
    def __init__(self):
        super(ObjectnessClusteringLossConfig, self).__init__()
        self.threshold = 1  # Should match the threshold in model config file
        self.center_foreground = False
        self.scaleable = True


class MaskAndMassLossConfig(ObjectnessClusteringLossConfig):
    def __init__(self):
        super(MaskAndMassLossConfig, self).__init__()
        self.mass_loss_weight = .1
        self.instance_only = False


class ObjectnessRPNLossConfig(ObjectnessLossConfig):
    def __init__(self):
        super(ObjectnessRPNLossConfig, self).__init__()
        self.filter = False  # No filter implemented yet for this model
        self.roi_config = ROIModuleConfig()
        self.regression = False  # has to match the entry in RPNModelConfig
        self.deltas = [(0, 0, 0, 0),
                       (0, 0, 1, 0),
                       (0, 0, -1, 0),
                       (0, 0, 0, 1),
                       (0, 0, 0, -1),
                       (1, 0, 1, 0),
                       (-1, 0, -1, 0),
                       (0, 1, 0, 1),
                       (0, -1, 0, -1)]
        self.regression_weight = 1


class FgBgLossConfig:
    def __init__(self):
        self.prioritized_replay = False
        self.restrict_positives = False
        self.restrict_negatives = True and not self.restrict_positives
        self.kernel = actor_config.check_change_kernel
        self.kernel_size = (self.kernel.shape[0] - 1) // 2
        self.foreground_threshold = 1.5


class TrainerConfig:
    def __init__(self):
        self.log_path = None
        self.checkpoint_path = None
        self.save_frequency = 100
        self.ground_truth = 0  # 0 = self, 1 = poke, 2 = mask, 3 = poke+mask, 4 = visualize, 5 = generate test set
        self.num_actors = 35
        self.episodes = 900
        self.new_datapoints_per_episode = 70
        self.batch_size = 64
        self.lr_schedule = lambda episode, episodes: 5e-4
        self.weight_decay = 1e-4
        self.update_schedule = lambda episode, episodes: int(15 + 30 * episode / episodes)
        self.poking_schedule = lambda episode, episodes: 20
        self.prefill_memory = 3000
        self.eval_during_train = False
        self.unfreeze = -1


class TestingConfig:
    def __init__(self):
        self.num_actors = 25
        self.bs = 50
        self.colors = [(0, 0, 200), (0, 255, 255), (255, 0, 255), (255, 255, 0),
                       (120, 255, 0), (255, 120, 0), (0, 255, 120), (0, 120, 255), (120, 0, 255), (120, 255, 120),
                       (60, 177, 0), (177, 60, 0), (0, 177, 60), (0, 60, 177), (60, 0, 177), (60, 177, 60)]
