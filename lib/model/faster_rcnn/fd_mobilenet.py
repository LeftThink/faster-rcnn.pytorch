from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn_cascade import _fasterRCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import math
import pdb

__all__ = [
    'fd_mobilenet',
]

_mobilenet_fast_downsampling_channels = [64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 1024]
_mobilenet_fast_downsampling_strides = [2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1]

class MobileNet(nn.Module):
    def __init__(self, init_features, channels, strides):
        super(MobileNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv_0', nn.Conv2d(3, init_features, 3, stride=2, padding=1, bias=False)),
            ('norm_0', nn.BatchNorm2d(init_features)),
            ('relu_0', nn.ReLU(inplace=True)),
        ]))
        in_c = init_features
        for _, (out_c, stride) in enumerate(zip(channels, strides)):
            self.features.add_module('dw_conv_{}'.format(_), nn.Conv2d(in_c, in_c, 3, stride=stride, padding=1, groups=in_c, bias=False))
            self.features.add_module('dw_norm_{}'.format(_), nn.BatchNorm2d(in_c))
            self.features.add_module('dw_relu_{}'.format(_), nn.ReLU(inplace=True))
            self.features.add_module('pw_conv_{}'.format(_), nn.Conv2d(in_c, out_c, 1, bias=False))
            self.features.add_module('pw_norm_{}'.format(_), nn.BatchNorm2d(out_c))
            self.features.add_module('pw_relu_{}'.format(_), nn.ReLU(inplace=True))
            in_c = out_c
        self.pool = nn.AvgPool2d(7)
        self.classifier = nn.Linear(in_c, 1000)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def _fast_downsampling_mobilenet(width_mul):
    init_features = int(32 * width_mul)
    channels = [int(x * width_mul) for x in _mobilenet_fast_downsampling_channels]
    return MobileNet(init_features, channels, _mobilenet_fast_downsampling_strides)

class fd_mobilenet(_fasterRCNN):
    def __init__(self, wm, classes, pretrained=False, class_agnostic=False):
        self.wm = wm
        if wm == 1:
            self.model_path = 'data/pretrained_model/fd-mobilenet-1x.pth.tar'
            self.dout_base_model = 512
            self.dout_top_model = 1024
        elif wm == 0.5:
            self.model_path = 'data/pretrained_model/fd-mobilenet-0.5x.pth.tar'
            self.dout_base_model = 256
            self.dout_top_model = 512
        elif wm == 0.25:
            self.model_path = 'data/pretrained_model/fd-mobilenet-0.25x.pth.tar'
            self.dout_base_model = 128
            self.dout_top_model = 256
        else:
            raise ValueError("Width Multiplier not supported!.")
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic

        _fasterRCNN.__init__(self, classes, class_agnostic)

    def _init_modules(self):
        mobilenet = _fast_downsampling_mobilenet(self.wm)
        if self.pretrained == True:
            print("Loading pretrained weights from %s" %(self.model_path))
            checkpoint = torch.load(self.model_path)
            mobilenet.load_state_dict(checkpoint['state_dict'])

        self.RCNN_base = nn.Sequential(*list(mobilenet.features.children())[:57])
        self.RCNN_top = nn.Sequential(*list(mobilenet.features.children())[57:69])

        self.RCNN_cls_score = nn.Linear(self.dout_top_model, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(self.dout_top_model, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(self.dout_top_model, 4 * self.n_classes)

        for p in self.RCNN_base[0].parameters(): p.requires_grad=False

        assert (0 <= cfg.FD_MOBILENET.FIXED_LAYERS < 12)
        if cfg.FD_MOBILENET.FIXED_LAYERS >= 1:
            for _ in range(3):
                for p in self.RCNN_base[_].parameters(): p.requires_grad=False
        if cfg.FD_MOBILENET.FIXED_LAYERS >= 2:
            for _ in range(3, 3 + 6 * (cfg.FD_MOBILENET.FIXED_LAYERS - 1)):
                for p in self.RCNN_base[_].parameters(): p.requires_grad=False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad=False

        self.RCNN_base.apply(set_bn_fix)
        self.RCNN_top.apply(set_bn_fix)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            self.RCNN_base.eval()
            for _ in range(57):
                self.RCNN_base[_].train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.RCNN_base.apply(set_bn_eval)
            self.RCNN_top.apply(set_bn_eval)

    def _head_to_tail(self, pool5):
        fc7 = self.RCNN_top(pool5).mean(3).mean(2)
        return fc7
