#!/usr/bin/env python

"""
    yolo_worker.py
"""
import sys
import io
import numpy as np
import os
import logging

from PIL import Image

from .base import BaseWorker
from tdesc.artifacts import YoloFeature


def _import_yolo():
    global DarknetObjectDetector
    from libpydarknet import DarknetObjectDetector


class DetBBox(object):

    def __init__(self, bbox):
        self.left = bbox.left
        self.right = bbox.right
        self.top = bbox.top
        self.bottom = bbox.bottom
        self.confidence = bbox.confidence
        self.cls = bbox.cls


class YoloWorker(BaseWorker):

    def __init__(self, cfg_path, weight_path, name_path,
                 thresh=0.1, nms=0.3, target_dim=416,
                 logger=None, *args, **kwargs):
        _import_yolo()

        self.logger = logger or logging

        DarknetObjectDetector.set_device(int(os.environ.get("CUDA_VISIBLE_DEVICES", "")))

        self.target_dim = target_dim
        self.class_names = open(name_path).read().splitlines()
        self.det = DarknetObjectDetector(cfg_path, weight_path, thresh, nms, 0)

        self.logger.debug('YoloWorker: ready')

    def _detection_message(self, yolo_artifact):
        return "Yolo Objects Detected: {}".format(yolo_artifact.filepath)

    def imread(self, path):

        img = Image.open(path).convert('RGB')
        img = img.resize((self.target_dim, self.target_dim), Image.BILINEAR)


        data = np.array(img).transpose([2,0,1]).astype(np.uint8).tostring()

        return data, (img.size[0], img.size[1])

    def featurize(self, yolo_artifact, obj):

        data, size = obj

        bboxes = [
            DetBBox(x)
            for x in self.det.detect_object(str(data), size[0], size[1], 3).content
        ]

        feats = []

        for bbox in bboxes:
            yolo_artifact.features.append(
                YoloFeature(
                    self.class_names[bbox.cls],
                    bbox.confidence,
                    [bbox.top, bbox.bottom, bbox.left, bbox.right],
                )
            )

        return yolo_artifact
