import cv2
import random
import numpy as np

from utils.auxiliary_processing import random_range
from visualizer.visual_image import visual_image


class Translate:
    def __init__(self,
                 dx,
                 dy,
                 background=(0,0,0)):
        self.dx = dx
        self.dy = dy
        self.background = background

    def __call__(self, image, mask):
        h, w, _ = image.shape
        dx_abs = int(round(w * self.dx)) if isinstance(self.dx, float) else self.dx
        dy_abs = int(round(h * self.dy)) if isinstance(self.dy, float) else self.dy
        M = np.float32([[1, 0, dx_abs],
                        [0, 1, dy_abs]])
        image = cv2.warpAffine(image,
                               M=M,
                               dsize=(w, h),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=self.background)
        mask = cv2.warpAffine(mask,
                               M=M,
                               dsize=(w, h),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=self.background)
        return image, mask
      

class RandomTranslate:
    def __init__(self, 
                 dx_minmax=(0.03, 0.3),
                 dy_minmax=(0.03, 0.3),
                 background=(0,0,0),
                 prob=0.5):
        self.dx_minmax  = dx_minmax
        self.dy_minmax  = dy_minmax
        self.prob       = prob
        self.aug        = Translate(dx=0,
                                    dy=0,
                                    background=background)
        
    def __call__(self, image, mask):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            dy_abs = np.random.uniform(self.dy_minmax[0], self.dy_minmax[1])
            dx_abs = np.random.uniform(self.dx_minmax[0], self.dx_minmax[1])
            dy = np.random.choice([-dy_abs, dy_abs])
            dx = np.random.choice([-dx_abs, dx_abs])
            self.aug.dx = dx
            self.aug.dy = dy
            image, mask = self.aug(image, mask)
        return image, mask
    
    
if __name__ == "__main__":
    image_path = "/home/vbpo/Desktop/TuNIT/working/Plate Recognitions/repo1/images/93def03e-1732-478e-8c83-3ffd1bad1c42_3.jpg"
    mask_path  = "/home/vbpo/Desktop/TuNIT/working/Plate Recognitions/repo1/images/93def03e-1732-478e-8c83-3ffd1bad1c42_3.png"
    image      = cv2.imread(image_path)
    mask       = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    visual_image([np.array(image).astype(np.float32)/255.0, np.array(mask).astype(np.float32)/255.0], ['original image', 'original mask'])

    augment1 = Translate(dx=-50, dy=0.05)
    images1, mask1 = augment1(image, mask)
    visual_image([images1, np.array(mask1).astype(np.float32)/255.0], ['flip image', 'flip mask'])
    tensor_value_info(images1)
    tensor_value_info(mask1)

    augment2 = RandomTranslate(dx_minmax=(-0.5, 0.5),
                               dy_minmax=(-0.5, 0.5),
                               background=(0,0,0),
                               prob=0.9)
    images2, mask2 = augment2(image, mask)
    visual_image([np.array(images2).astype(np.float32)/255.0, np.array(mask2).astype(np.float32)/255.0], ['flip image', 'flip mask'])