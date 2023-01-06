import cv2
import math
import random
import numpy as np

from utils.auxiliary_processing import random_range
from visualizer.visual_image import visual_image


class Rotate:
    def __init__(self,
                 angle):
        self.angle = angle

    def __call__(self, image, mask):
        h, w, _ = image.shape
        center_coord = (w / 2, h / 2)
        
        rot = cv2.getRotationMatrix2D(center_coord, self.angle, 1)
        rad = math.radians(self.angle)
        sin = math.sin(rad)
        cos = math.cos(rad)
        b_w = int((h * abs(sin)) + (w * abs(cos)))
        b_h = int((h * abs(cos)) + (w * abs(sin)))
        rot[0, 2] += ((b_w / 2) - center_coord[0])
        rot[1, 2] += ((b_h / 2) - center_coord[1])
        image = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
        mask  = cv2.warpAffine(mask, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
        return image, mask
    
    
class RandomRotate:
    def __init__(self, 
                 angle_range=(-30, 30),
                 prob=0.5):
        self.angle_range  = angle_range
        self.prob       = prob
        self.aug        = Rotate(angle=0)
        
    def __call__(self, image, mask):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            angle_abs = np.random.uniform(self.angle_range[0], self.angle_range[1])
            angle = np.random.choice([-angle_abs, angle_abs])
            self.aug.angle = angle
            image, mask = self.aug(image, mask)
        return image, mask
    

if __name__ == "__main__":
    image_path = "/home/vbpo/Desktop/TuNIT/working/Plate Recognitions/repo1/images/93def03e-1732-478e-8c83-3ffd1bad1c42_3.jpg"
    mask_path  = "/home/vbpo/Desktop/TuNIT/working/Plate Recognitions/repo1/images/93def03e-1732-478e-8c83-3ffd1bad1c42_3.png"
    image      = cv2.imread(image_path)
    mask       = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    visual_image([np.array(image).astype(np.float32)/255.0, np.array(mask).astype(np.float32)/255.0], ['original image', 'original mask'])
    tensor_value_info(image)
    augment1 = Rotate(angle=30)
    images1, mask1 = augment1(image, mask)
    visual_image([images1, np.array(mask1).astype(np.float32)/255.0], ['flip image', 'flip mask'])
    tensor_value_info(images1)
    tensor_value_info(mask1)

    augment2 = RandomRotate(angle_range=(-30, 30),
                            prob=1)
    images2, mask2 = augment2(image, mask)
    visual_image([np.array(images2).astype(np.float32)/255.0, np.array(mask2).astype(np.float32)/255.0], ['flip image', 'flip mask'])