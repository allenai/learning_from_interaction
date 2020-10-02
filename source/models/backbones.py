import torch
from torch import nn
# from detectron2 import model_zoo
# from detectron2.config import get_cfg
# from detectron2.modeling import build_backbone

from config import BackboneConfig, global_config
from models.model_utils import ResBlock, UpConv


class UNetBackbone(nn.Module):
    def __init__(self, model_config: BackboneConfig):
        super(UNetBackbone, self).__init__()
        self.config = model_config

        self.conv1 = nn.Sequential(nn.Conv2d(3 + global_config.depth, 32,
                                             kernel_size=5, stride=3, padding=2, bias=False),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU())
        if model_config.small:
            self.block1 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=2, bias=False),
                                        nn.BatchNorm2d(64), nn.ReLU())
            self.block2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2, bias=False),
                                        nn.BatchNorm2d(128), nn.ReLU())
            self.block3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=2, bias=False),
                                        nn.BatchNorm2d(256), nn.ReLU())
        else:
            self.block1 = ResBlock(32, first_kernel_size=3)
            self.block2 = ResBlock(64, first_kernel_size=3)
            self.block3 = ResBlock(128, first_kernel_size=3)

        self.upconv1 = UpConv(256, 128, 128)
        self.upconv2 = UpConv(128, 64, 64)
        self.upconv3 = UpConv(64, 32, 64)

    def forward(self, x):
        """
        :param x: Shape BS x 3(+1) x resolution x resolution
        :return: (Shape BS x D x grid_size x grid_size, BS x grid_size x grid_size), BS x D' (optional)
        """
        x = self.conv1(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        y3 = self.upconv1(x3, x2)
        y2 = self.upconv2(y3, x1)
        y1 = self.upconv3(y2, x)
        y = torch.nn.functional.interpolate(y1, size=(global_config.grid_size, global_config.grid_size),
                                            mode='bilinear')

        return y, x3.mean(dim=(2, 3)).unsqueeze(2).unsqueeze(3)


# class R50FPNBackbone(nn.Module):
#     def __init__(self):
#         super(R50FPNBackbone, self).__init__()
#         self.backbone = build_backbone(make_rpn50_fpn_config())
#
#     def forward(self, x):
#         x = torch.nn.functional.interpolate(x, size=(800, 800), mode='bilinear')
#         y = self.backbone(x)
#         y_filter = y['p6'].mean(dim=(2, 3)).unsqueeze(2).unsqueeze(3)
#         y = y['p3']
#         if global_config.grid_size != 100:
#             y = torch.nn.functional.interpolate(y, size=(global_config.grid_size,
#                                                          global_config.grid_size), mode='bilinear')
#         return y, y_filter
#
#
# def make_rpn50_fpn_config():
#     cfg = get_cfg()
#     cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
#     cfg.MODEL.PIXEL_MEAN = cfg.MODEL.PIXEL_MEAN + [1.5] * global_config.depth
#     cfg.MODEL.PIXEL_STD = cfg.MODEL.PIXEL_STD + [1.] * global_config.depth
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
#     cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = True
#     return cfg
