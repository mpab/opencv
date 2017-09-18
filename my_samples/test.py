import cv2  # The opencv python bindings
import numpy  # Python scienticic computing library

import abc   # abstract base classes
from typing import List  # type annotations
from utils.cropengine import Feature  # I'll explain this one soon.

FileName = str  # type alias
# OpenCv represents all images as n-dimensional numpy arrays.
# For clarity and convenience, we'll just call it "CvImage"
CVImage = numpy.ndarray

class FeatureDetector(abc.ABC):
    """Abstract base class for the feature detectors."""
    
    @abc.abstractmethod
    def __init__(self, n:int, size:int) -> None:
        ...

    @abc.abstractmethod
    def detect_features(self, fn: FileName) -> List[Feature]:
        """Find the most salient features of the image."""

def opencv_image(fn: str, resize: int=0) -> CVImage:
    """Read image file to grayscale openCV int array.

    The OpenCV algorithms works on a two dimensional
    numpy array integers where 0 is black and 255 is
    white. Color images will be converted to grayscale.
    """
    cv_image = cv2.imread(fn)
    cv_image = cv2.cvtColor(
        cv_image, cv2.COLOR_BGR2GRAY)
    if resize > 0:
        w, h = cv_image.shape[1::-1]  # type: int, int
        multiplier = (resize ** 2 / (w * h)) ** 0.5
        dimensions = tuple(
            int(round(d * multiplier)) for d in (w, h))
        cv_image = cv2.resize(
            cv_image, dimensions,
            interpolation=cv2.INTER_AREA)
    return cv_image

def resize_feature(feature: Feature, cv_image: CVImage) -> Feature:
    """Convert a Feature to a relative coordinate system.

    The output will be in a normalized coordinate system
    where the image width and height are both 1.
    Any part of the Feature that overflows the image
    frame will be truncated.
    """
    img_h, img_w = cv_image.shape[:2]  # type: int, int
    feature = Feature(
        label=feature.label,
        weight=feature.weight / (img_w * img_h),
        left=max(0, feature.left / img_w),
        top=max(0, feature.top / img_h),
        right=min(1, feature.right / img_w),
        bottom=min(1, feature.bottom / img_h),
    )
    return feature

monkey_race = './monkey-race-wide.jpg'
show_image(monkey_race)