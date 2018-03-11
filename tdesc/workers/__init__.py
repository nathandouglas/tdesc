from __future__ import print_function, absolute_import

import sys

from .base import BaseWorker

try:
    from .dlib_worker import DlibFaceWorker
    from .dlib_batch_worker import DlibFaceBatchWorker
except:
    print('cannot load dlib workers')

from .yolo_worker import YoloWorker
from .vgg16_worker import VGG16Worker
