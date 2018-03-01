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
import urllib


from .base import BaseWorker


def import_vgg16():
    global VGG16
    global Model
    global load_model
    global image
    global preprocess_input
    global K

    from keras.applications import VGG16
    from keras.models import Model, load_model
    from keras.preprocessing import image
    from keras.applications.vgg16 import preprocess_input

    from keras import backend as K


class VGG16Worker(BaseWorker):
    """
        compute late VGG16 features

        either densely connected (default) or crow (sum-pooled conv5)
    """
    def __init__(self, crow, target_dim=224, logger=None, model_path=None):

        try:
            import_vgg16()
        except Exception as e:
            # TODO handle non GPU machines
            assert e == 'a'

        if K.backend() == 'tensorflow':
            self._set_session()

        self.crow = crow
        self.model_path = model_path
        self.model = self._get_model()
        self.target_dim = target_dim
        self._warmup()
        self.logger = logger or logging

        if self.logger:
            self.logger.info(
                'VGG16Worker: ready (target_dim=%d)' % ( int(target_dim) )
            )

    def _get_model(self):
        model = None

        if self.crow:
            model = VGG16(weights='imagenet', include_top=False)

        elif self.model_path:
            model = load_model(self.model_path)

        else:
            whole_model = VGG16(weights='imagenet', include_top=True)
            model = Model(inputs=whole_model.input, outputs=whole_model.get_layer('fc2').output)

        return model

    def _set_session(self):

        cfg = K.tf.ConfigProto()
        cfg.gpu_options.allow_growth = True

        # cfg.gpu_options.visible_device_list = os.environ.get('GPU_DEV', '0')

        cfg.gpu_options.per_process_gpu_memory_fraction = float(
            os.environ.get('GPU_MEMORY_FRACTION', '0.1')
        )

        self.sess = K.tf.Session(config=cfg)

        K.set_session(self.sess)

    def _warmup(self):
        self.model.predict(np.zeros((1, self.target_dim, self.target_dim, 3)))

    def imread(self, path):

        if path[:4] == 'http':
            with contextlib.closing(urllib.request.urlopen(path)) as req:
                path = io.StringIO(req.read())

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
