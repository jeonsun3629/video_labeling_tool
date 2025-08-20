"""
분류기 모듈
"""
from .dinov2_classifier import DINOv2Classifier
from .clip_classifier import CLIPClassifier
from .universal_classifier import UniversalClassifier

__all__ = ['DINOv2Classifier', 'CLIPClassifier', 'UniversalClassifier']
