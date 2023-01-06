import cv2
import random
import numpy as np

from visualizer.visual_image import visual_image


class Resize:
    def __init__(self,
                 target_size,
                 interpolation=cv2.INTER_NEAREST):
        self.target_size = target_size
        self.interpolation = interpolation

    def __call__(self, image, mask):
        image = cv2.resize(image, self.target_size[:2], interpolation=self.interpolation)
        mask  = cv2.resize(mask, self.target_size[:2], interpolation=self.interpolation)
        return image, mask
    
if __name__ == "__main__":
    image_path = "/home/vbpo/Desktop/TuNIT/working/Plate Recognitions/repo1/images/93def03e-1732-478e-8c83-3ffd1bad1c42_3.jpg"
    mask_path  = "/home/vbpo/Desktop/TuNIT/working/Plate Recognitions/repo1/images/93def03e-1732-478e-8c83-3ffd1bad1c42_3.png"
    image      = cv2.imread(image_path)
    mask       = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    visual_image([np.array(image).astype(np.float32)/255.0, np.array(mask).astype(np.float32)/255.0], ['original image', 'original mask'])
    tensor_value_info(image)
    augment1 = Resize(target_size=(10, 50))
    images1, mask1 = augment1(image, mask)
    visual_image([images1, np.array(mask1).astype(np.float32)/255.0], ['flip image', 'flip mask'])
    tensor_value_info(images1)
    tensor_value_info(mask1)