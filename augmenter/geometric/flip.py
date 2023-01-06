import cv2
import random
import numpy as np

from utils.auxiliary_processing import random_range
from visualizer.visual_image import visual_image


class Flip:
    def __init__(self, mode='horizontal'):
        self.mode = mode

    def __call__(self, image, mask):
        h, w, _ = image.shape
        horizontal_list = ['horizontal', 'h']
        vertical_list   = ['vertical', 'v']
        if self.mode.lower() in horizontal_list:
            image = image[:,::-1]
            mask  = mask[:,::-1]
        elif self.mode.lower() in vertical_list:
            image = image[::-1]
            mask  = mask[::-1]

        return image, mask
      

class RandomFlip:
    def __init__(self, prob=0.5, mode='horizontal'):
        self.prob       = prob
        self.mode       = mode
        self.aug        = Flip(mode=self.mode)
        
    def __call__(self, image, mask):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            image, mask = self.aug(image, mask)
        return image, mask
      

if __name__ == "__main__":
    image_path = "/content/sample_data/voc_tiny/train/000288.jpg"
    mask_path  = "/content/sample_data/voc_tiny/train/000288.png"
    image      = cv2.imread(image_path)
    mask       = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    visual_image([np.array(image).astype(np.float32)/255.0, np.array(mask).astype(np.float32)/255.0], ['original image', 'original mask'], size=(20, 20))

    augment1 = Flip(mode='horizontal')
    images1, mask1 = augment1(image, mask)
    visual_image([np.array(images1).astype(np.float32)/255.0, np.array(mask1).astype(np.float32)/255.0], ['flip image', 'flip mask'], size=(20, 20))

    augment2 = RandomFlip(prob=0.5, mode='horizontal')
    images2, mask2 = augment2(image, mask)
    visual_image([np.array(images2).astype(np.float32)/255.0, np.array(mask2).astype(np.float32)/255.0], ['flip image', 'flip mask'], size=(20, 20))