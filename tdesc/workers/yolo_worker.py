#!/usr/bin/env python

"""
    yolo_worker.py
"""

import sys
import urllib.request, urllib.parse, urllib.error
import contextlib
import io
import numpy as np
from PIL import Image

from .base import BaseWorker


class DetBBox(object):

    def __init__(self, bbox):
        self.left = bbox.left
        self.right = bbox.right
        self.top = bbox.top
        self.bottom = bbox.bottom
        self.confidence = bbox.confidence
        self.cls = bbox.cls


class YoloWorker(BaseWorker):

    def __init__(self, cfg_path, weight_path, name_path, thresh=0.1, nms=0.3, target_dim=416):
        self._import_yolo()

        DarknetObjectDetector.set_device(0)
        self.target_dim = target_dim
        self.class_names = open(name_path).read().splitlines()
        self.det = DarknetObjectDetector(cfg_path, weight_path, thresh, nms, 0)
        print('YoloWorker: ready', file=sys.stderr)

    def _import_yolo(self):
        global DarknetObjectDetector
        from libpydarknet import DarknetObjectDetector

    def imread(self, path):

        if path[:4] == 'http':
            with contextlib.closing(urllib.request.urlopen(path)) as req:
                path = io.StringIO(req.read())

        img = Image.open(path).convert('RGB')
        img = img.resize((self.target_dim, self.target_dim), Image.BILINEAR)

        data = np.array(img).transpose([2,0,1]).astype(np.uint8).tostring()

        return data, (img.size[0], img.size[1])

    def featurize(self, yolo_artifact, obj):

        data, size = obj

        bboxes = [DetBBox(x) for x in self.det.detect_object(data, size[0], size[1], 3).content]
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
