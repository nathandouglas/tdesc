#!/usr/bin/env python

"""
    vgg16_worker.py
"""
import contextlib
import io
import logging
import os
import numpy as np
import sys
import urllib.request, urllib.parse, urllib.error

from keras                    import backend as K
from keras.applications       import VGG16
from keras.models             import Model
from keras.preprocessing      import image
from keras.applications.vgg16 import preprocess_input

from .base import BaseWorker


class VGG16Worker(BaseWorker):
    """
        compute late VGG16 features

        either densely connected (default) or crow (sum-pooled conv5)
    """
    def __init__(self, crow, target_dim=224, logger=None):
        if K.backend() == 'tensorflow':
            self._limit_mem()

        if crow:
            self.model = VGG16(weights='imagenet', include_top=False)
        else:
            whole_model = VGG16(weights='imagenet', include_top=True)
            self.model = Model(inputs=whole_model.input, outputs=whole_model.get_layer('fc2').output)

        self.target_dim = target_dim
        self.crow = crow

        self._warmup()

        self.logger = logger or logging

        if self.logger:
            self.logger.info(
                'VGG16Worker: ready (target_dim=%d)' % ( int(target_dim) )
            )

    def _limit_mem(self):
        cfg = K.tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        cfg.gpu_options.visible_device_list = os.environ.get('GPU_DEV', '0')
        cfg.gpu_options.per_process_gpu_memory_fraction = float(os.environ.get('GPU_MEMORY_FRACTION', '0.1'))

        self.sess = K.tf.Session(config=cfg)

        K.set_session(self.sess)

    def _warmup(self):
        _ = self.model.predict(np.zeros((1, self.target_dim, self.target_dim, 3)))

    def imread(self, path):
        if 'http' == path[:4]:
            with contextlib.closing(urllib.request.urlopen(path)) as req:
                local_url = io.StringIO(req.read())
            img = image.load_img(local_url, target_size=(self.target_dim, self.target_dim))
        else:
            img = image.load_img(path, target_size=(self.target_dim, self.target_dim))

        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        return img

    def featurize(self, image_artifact, img):
        feat = self.model.predict(img).squeeze()

        if self.crow:
            feat = feat.sum(axis=(0, 1))

        return (image_artifact, feat)
