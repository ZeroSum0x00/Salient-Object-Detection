import cv2
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.utils import Sequence

from data_utils.data_processing import extract_data_folder, get_data, Normalizer
from data_utils.data_augmentation import Augmentor
from utils.logger import logger
from configs import general_config as cfg


def get_train_test_data(data_zipfile            = cfg.DATA_PATH, 
                        annotation_dir          = cfg.DATA_ANNOTATION_PATH,
                        dst_dir                 = cfg.DATA_DESTINATION_PATH,
                        target_size             = cfg.TARGET_SIZE, 
                        batch_size              = cfg.TRAIN_BATCH_SIZE, 
                        augmentor               = cfg.DATA_AUGMENTATION,
                        normalizer              = cfg.DATA_NORMALIZER,
                        check_data              = cfg.CHECK_DATA, 
                        load_memory             = cfg.DATA_LOAD_MEMORY,
                        *args, **kwargs):
    
    data_folder = extract_data_folder(data_zipfile, dst_dir)
    data_train = get_data(data_folder,
                          annotation_dir    = annotation_dir,
                          phase             = 'train', 
                          check_data        = check_data,
                          load_memory       = load_memory)
                          
    train_generator = Train_Data_Sequence(data_train, 
                                          target_size             = target_size, 
                                          batch_size              = batch_size, 
                                          augmentor               = augmentor['train'],
                                          normalizer              = normalizer,
                                          *args, **kwargs)

    data_valid = get_data(data_folder,
                          annotation_dir    = annotation_dir,
                          phase             = 'validation', 
                          check_data        = check_data,
                          load_memory       = load_memory)
    valid_generator = Valid_Data_Sequence(data_valid, 
                                          target_size             = target_size, 
                                          batch_size              = batch_size, 
                                          augmentor               = augmentor['validation'],
                                          normalizer              = normalizer,
                                          *args, **kwargs)
    
    logger.info('Load data successfully')
    return train_generator, valid_generator


class Train_Data_Sequence(Sequence):
    def __init__(self, 
                 dataset, 
                 target_size             = cfg.TARGET_SIZE,
                 batch_size              = cfg.TRAIN_BATCH_SIZE,
                 normalizer              = cfg.DATA_NORMALIZER,
                 augmentor               = cfg.DATA_AUGMENTATION['train']):
        
        self.data_path = dataset['data_path']
        self.dataset = dataset['data_extractor']
        if isinstance(augmentor, list):
            self.augmentor = Augmentor(augment_objects=augmentor)
        else:
            self.augmentor = augmentor

        self.target_size = target_size
        self.batch_size = batch_size

        self.dataset = shuffle(self.dataset)

        self.N = self.n = len(self.dataset)
        self.normalizer = Normalizer(normalizer)

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, index):
        batch_image = []
        batch_mask  = []

        for i in range(index * self.batch_size, (index + 1) * self.batch_size):  
            i           = i % self.N
            sample = self.dataset[i]
            img_path = self.data_path + sample['image_name']
            image = cv2.imread(img_path)
            mask_path = self.data_path + sample['mask_name']
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if self.augmentor:
                image, mask  = self.augmentor(image, mask)

            image, mask = self.normalizer(image, 
                                          mask,
                                          target_size=self.target_size,
                                          resize_mode='inflexible',
                                          interpolation=cv2.INTER_NEAREST)
            mask = np.expand_dims(mask, axis=-1)
            batch_image.append(image)
            batch_mask.append(mask)

        batch_image = np.array(batch_image)
        batch_mask  = np.array(batch_mask)
        return batch_image, batch_mask
        
    def on_epoch_end(self):
        self.dataset = shuffle(self.dataset)

class Valid_Data_Sequence(Sequence):
    def __init__(self, 
                 dataset, 
                 target_size             = cfg.TARGET_SIZE,
                 batch_size              = cfg.TRAIN_BATCH_SIZE,
                 normalizer              = cfg.DATA_NORMALIZER,
                 augmentor               = cfg.DATA_AUGMENTATION['train']):
        
        self.data_path = dataset['data_path']
        self.dataset = dataset['data_extractor']
        
        if isinstance(augmentor, list):
            self.augmentor = Augmentor(augment_objects=augmentor)
        else:
            self.augmentor = augmentor
            
        self.target_size = target_size
        self.batch_size = batch_size

        self.N = self.n = len(self.dataset)
        self.normalizer = Normalizer(normalizer)

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, index):
        batch_image = []
        batch_mask  = []
        
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):  
            i           = i % self.N
            sample = self.dataset[i]
            img_path = self.data_path + sample['image_name']
            image = cv2.imread(img_path)
            mask_path = self.data_path + sample['mask_name']
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if self.augmentor:
                image, mask  = self.augmentor(image, mask)

            image, mask = self.normalizer(image, 
                                          mask,
                                          target_size=self.target_size,
                                          resize_mode='inflexible',
                                          interpolation=cv2.INTER_NEAREST)
            mask = np.expand_dims(mask, axis=-1)
            batch_image.append(image)
            batch_mask.append(mask)

        batch_image = np.array(batch_image)
        batch_mask  = np.array(batch_mask)
        return batch_image, batch_mask