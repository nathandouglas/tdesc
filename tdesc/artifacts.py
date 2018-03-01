"""
Centralized models module.

Contains all model objects used within
the ApeFace analytics.
"""
import os
import h5py
import json

from datetime import datetime


class Feature(object):
    def __init__(self, error='', *args, **kwargs):
        self.created = datetime.utcnow()
        self.error = error

    def to_dict(self):
        return {
            "created": str(self.created),
            "error": self.error
        }

class ImageArtifact(object):
    """ImageArtifact class.

    Data structure used to communicate between
    Image analytics, and application layer.
    """
    def __init__(self, id, filepath, *args, **kwargs):
        self.id = id
        self.filepath = filepath
        self.db = self._create_db()
        self.image = None

    def _create_db(self):
        """Creates a DB (h5) file.

        If existing DB exists, delete it
        and create a new one.

        Returns:
            h5py file
        """
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

        return h5py.File(self.db_path)

    @property
    def db_path(self):
        """Returns db path to save h5."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, dic, *args, **kwargs):
        """Return an ImageArtifact from dict

        Method assumes dict to have the following
        minimum structure:
            {
                '_id': 'string',
                'md5': 'string',
                'filepath': 'string',
                ...
            }

        Args:
            dic: dict, see above

        Returns:
            ImageArtifact
        """
        return cls(dic['_id'], dic['filepath'], *args, **kwargs)

    def to_dict(self):
        """Returns dict representation of class."""
        return {"_id": self.id, "filepath": filepath}


class YoloFeature(Feature):
    def __init__(self, class_name, confidence, bbox, *args, **kwargs):
        super(YoloFeature, self).__init__(*args, **kwargs)

        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox

    def to_dict(self):
        base = super(YoloFeature, self).to_dict()

        base.update({
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox
        })

        return base


class YoloArtifact(ImageArtifact):
    def __init__(self, *args, **kwargs):
        super(YoloArtifact, self).__init__(*args, **kwargs)

        self.features = []

    def _create_db(self):
        return None

    def to_dict(self):
        return {
            "_id": self.id,
            "yolo": [yf.to_dict() for yf in self.features]
        }
