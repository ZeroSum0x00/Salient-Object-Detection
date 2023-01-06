# import config parameters
from augmenter.augmentation import basic_augmenter

DATA_PATH = "/home/vbpo/Desktop/TuNIT/working/Datasets/DUTS/"
# DATA_PATH = "/home/vbpo/Desktop/TuNIT/working/Plate Recognitions/repo1/yolov5/results"
DATA_ANNOTATION_PATH = None
DATA_DESTINATION_PATH = None
TARGET_SIZE = (320, 320, 3)
TRAIN_BATCH_SIZE = 8
DATA_AUGMENTATION               = basic_augmenter

DATA_NORMALIZER = 'divide'
CHECK_DATA = True
DATA_LOAD_MEMORY = False
TRAIN_EPOCH_INIT = 0
TRAIN_EPOCH_END = 500
TRAIN_WEIGHT_TYPE = None
TRAIN_WEIGHT_OBJECTS            = [        
                                    {
                                      'path': './saved_weights/20221114-133521/best_weights_mAP',
                                      'stage': 'full',
                                      'custom_objects': None
                                    }
                                  ]

TRAIN_RESULT_SHOW_FREQUENCY = 10

TRAIN_SAVE_WEIGHT_FREQUENCY     = 50

TRAIN_SAVED_PATH = "./saved_weights/"

TRAIN_MODE = 'graph'
