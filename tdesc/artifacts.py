"""
Centralized models module.

Contains all model objects used within
the ApeFace analytics.
"""
import os
import h5py
import json


class Feature(object):
    def to_dict(self):
        raise NotImplementedError

class FaceFeatures(Feature):
    """FaceFeature class.

    Encapsulates the results from dlib face
    detection.
    """
    def __init__(self, k, bbox, face_descriptor, chip_path=''):
        self.k = k
        self.bbox = bbox
        self.face_descriptor = face_descriptor
        self.chip_path = chip_path

    @property
    def has_face(self):
        """Check for face_descriptors.

        If any face_descriptors are found,
        then a face was detected.

        Returns:
            bool
        """
        return self.face_descriptor is not None

    def to_dict(self):
        """Return dict representation of FaceFeatures."""
        return {
            'k': self.k,
            'bbox': self.bbox.tolist(),
            'has_face': self.has_face,
            'chip_path': self.chip_path
        }

class ImageArtifact(object):
    """ImageArtifact class.

    Data structure used to communicate between
    Image analytics, and application layer.
    """
    def __init__(self, id, filepath):
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
    def from_dict(cls, dic):
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
        return cls(dic['_id'], dic['filepath'])

    def to_dict(self):
        """Returns dict representation of class."""
        return {"_id": self.id, "filepath": filepath}

class CrowFeatures(Feature):
    pass

class CrowArtifact(ImageArtifact):
    def __init__(self, id, filepath, *args, **kwargs):
        super(CrowArtifact, self).__init__(id, filepath, *args, **kwargs)

    @property
    def db_path(self):
        return '/{}/{}-crow.h5'.format(
            os.environ.get('H5S_DIR', '/h5s'),
            self.id
        )

    def to_dict(self):
        """Returns dict representation of class. """

        return {
            '_id': self.id,
            'crow': [ff.to_dict() for ff in self.faces],
        }

class FaceArtifact(ImageArtifact):
    """FaceArtifact class.

    Data structure used to communicate between
    face analytics, and application layer.
    """
    def __init__(self, id, filepath, *args, **kwargs):
        super(FaceArtifact, self).__init__(id, filepath, *args, **kwargs)
        self.faces = []

    @classmethod
    def from_dict(cls, dic):
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
        return cls(dic['_id'], dic['filepath'])

    @property
    def db_path(self):
        return '/{}/{}-face.h5'.format(
            os.environ.get('H5S_DIR', '/h5s'),
            self.id
        )

    @property
    def has_face(self):
        """Determine if any faces were detected.

        Check all faces to determine if any faces
        contain face descriptors.

        Returns:
            bool
        """
        return any(
            ff.has_face
            for ff in self.faces
        )

    def to_dict(self):
        """Returns dict representation of class. """

        return {
            '_id': self.id,
            'faces': [ff.to_dict() for ff in self.faces],
            'has_face': self.has_face
        }
