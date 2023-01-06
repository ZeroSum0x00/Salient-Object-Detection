import cv2
import numpy as np

from augmenter.geometric.resize import Resize
from augmenter.geometric.flip import RandomFlip
from augmenter.geometric.rotate import RandomRotate
from augmenter.geometric.translate import RandomTranslate



basic_augmenter = {
    'train': [
            RandomFlip(mode="horizontal"),
            RandomTranslate(dx_minmax=(-0.1, 0.1), dy_minmax=(-0.1, 0.1), background=(0,0,0)),
            RandomRotate(angle_range=(-30, 30)),
        ],
    'validation': [],
    'test': []
}