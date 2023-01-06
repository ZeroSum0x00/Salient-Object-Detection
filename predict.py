import os
import cv2
import logging
import numpy as np
import tensorflow as tf

from models import SOD
from models import U2Net
from data_utils.data_flow import get_train_test_data
from utils.files import get_files
from visualizer.visual_image import visual_image
from configs import general_config as cfg


def preprocess_input(image):
    image /= 255.0
    return image


def predict(image_path, model, input_shape):
    image = cv2.imread(image_path)
    ih, iw, _ = image.shape
    image = cv2.resize(image, input_shape[:2])
    image  = preprocess_input(image.astype(np.float32))
    image  = np.expand_dims(image, axis=0)
    pred = model.predict(image)
    pred = np.squeeze(pred, axis=0)
    pred = np.squeeze(pred, axis=-1)
    pred = cv2.resize(pred, (iw, ih))
    pred *= 255
    return pred

if __name__ == "__main__":
    input_shape = cfg.TARGET_SIZE

    architecture = U2Net(filters=[16, 32, 64, 128, 256, 512], classes=1)
    
    model = SOD(architecture=architecture, image_size=input_shape)

    weight_type = "weights"
    
    weight_objects = [        
                       {
                         'path': './saved_weights/20230104-132212/checkpoint_0100/saved_sod_weights',
                         'stage': 'full',
                         'custom_objects': None
                       }
                     ]
    
    if weight_type and weight_objects:
        if weight_type == "weights":
            model.load_weights(weight_objects)
        elif weight_type == "models":
            model.load_models(weight_objects)
        
    input_dir = "/home/vbpo/Desktop/TuNIT/working/Plate Recognitions/repo1/groundtruth/"
    image_names = get_files(input_dir, extensions=['jpg', 'jpeg'])

    for idx, name in enumerate(image_names):
        image_path = os.path.join(input_dir, name)
        mask = predict(image_path, model, input_shape)
        cv2.imwrite(f'./results/{name.replace(".jpg", ".png")}', mask)
    print("done!")