#!/usr/bin/env python

"""
    base.py

    !! Need to figure out how to handle timeouts, since sometimes this wants
    to be run on a stream coming from the internet that's slow or bursty
"""
import os
import sys
import json
import itertools
from time import time
from concurrent.futures import ThreadPoolExecutor

class BaseWorker(object):

    print_interval = 25

    def run(self, image_artifacts, io_threads=5, timeout=10, chunk_size=10000):
        start_time = time()
        pool = ThreadPoolExecutor(max_workers=io_threads)
        gen = (ia for ia in image_artifacts)
        total_images = len(image_artifacts)
        i = 0

        for chunk in self._chunker(gen, chunk_size):
            for image_artifact, imread_results in pool.map(self.do_io, chunk):

                i += 1

                self.logger.debug('{} of {}'.format(i, total_images))

                if imread_results is not None:

                    self.logger.debug(self._detection_message(image_artifact))

                    yield self.featurize(*(image_artifact, imread_results))

    def _chunker(self, iterable, chunk_size):
        while True:
            yield itertools.chain([next(iterable)], itertools.islice(iterable, chunk_size-1))

    def do_io(self, image_artifact):
        try:
            results = self.imread(image_artifact.filepath)
        except OSError:
            self.logger.error('Failed to read {}'.format(image_artifact.filepath))
            pass

        return (image_artifact, results)

