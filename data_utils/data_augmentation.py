from augmenter.augmentation import *
from configs import general_config as cfg


class Augmentor:
    def __init__(self, augment_objects):
        self.sequence_transform  = augment_objects

    def __call__(self, images, mask):
        if self.sequence_transform:
            for transform in self.sequence_transform:
                images, mask = transform(images, mask)
        return images, mask