import os
import cv2
import numpy as np

from utils.files import extract_zip, verify_folder, get_files
from data_utils.parse_annotations import ParseAnnotations


def extract_data_folder(data_dir, dst_dir=None):
    ACCEPTABLE_EXTRACT_FORMATS = ['.zip', '.rar', '.tar']
    if (os.path.isfile(data_dir)) and os.path.splitext(data_dir)[-1] in ACCEPTABLE_EXTRACT_FORMATS:
        if dst_dir is not None:
            data_destination = dst_dir
        else:
            data_destination = '/'.join(data_dir.split('/')[: -1])

        folder_name = data_dir.split('/')[-1]
        folder_name = os.path.splitext(folder_name)[0]
        data_destination = verify_folder(data_destination) + folder_name 

        if not os.path.isdir(data_destination):
            extract_zip(data_dir, data_destination)
        
        return data_destination
    else:
        return data_dir


def get_data(data_dir, 
             annotation_dir,
             phase, 
             check_data,
             load_memory,
             *args, **kwargs):
    data_dir = verify_folder(data_dir) + phase
    data_extraction = []
    
    image_files = sorted(get_files(data_dir, extensions=['png']))
    parser = ParseAnnotations(data_dir, annotation_dir, load_memory, check_data=check_data, *args, **kwargs)
    data_extraction = parser(image_files)

    dict_data = {
        'data_path': verify_folder(data_dir),
        'data_extractor': data_extraction
    }
    return dict_data


class Normalizer:
    def __init__(self, mode="divide"):
        self.mode = mode

    @classmethod
    def __get_standard_deviation(cls, img, mean=None, std=None):
        if mean is not None:
            for i in range(img.shape[-1]):
                if isinstance(mean, float) or isinstance(mean, int):
                    img[..., i] -= mean
                else:
                    img[..., i] -= mean[i]

                if std is not None:
                    for i in range(img.shape[-1]):
                        if isinstance(std, float) or isinstance(std, int):
                            img[..., i] /= (std + 1e-20)
                        else:
                            img[..., i] /= (std[i] + 1e-20)
        return img
    @classmethod
    def __resize_inflexible_mode(cls, image, mask, target_size, interpolation=None):
        h, w, _ = image.shape
        image_resized = cv2.resize(image, (target_size[1], target_size[0]), interpolation=interpolation)
        mask_resized = cv2.resize(mask, (target_size[1], target_size[0]), interpolation=interpolation)
        return image_resized, mask_resized

    @classmethod
    def __resize_flexible_mode(cls, image, mask, target_size, interpolation=None):
        h, w, _    = image.shape
        ih, iw, _  = target_size
        scale = min(iw/w, ih/h)
        nw, nh  = int(scale * w), int(scale * h)
        dw, dh = (iw - nw) // 2, (ih - nh) // 2
        image_resized = cv2.resize(image, (nw, nh))
        image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
        image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized

        mask_resized = cv2.resize(mask, (nw, nh))
        mask_paded = np.full(shape=[ih, iw, 3], fill_value=0.0)
        mask_paded[dh:nh+dh, dw:nw+dw, :] = mask_resized
        return image_paded, mask_paded

    def _sub_divide(self, image, mask, mean=None, std=None, target_size=None, resize_mode='inflexible', interpolation=None):
        if resize_mode.lower() == 'inflexible' and target_size:
            image, mask = self.__resize_inflexible_mode(image, mask, target_size, interpolation)
        elif resize_mode.lower() == 'flexible' and target_size:
            image, mask = self.__resize_flexible_mode(image, mask, target_size, interpolation)
        image = image.astype(np.float32)
        image = image / 127.5 - 1
        image = self.__get_standard_deviation(image, mean, std)
        image = np.clip(image, -1, 1)

        mask = mask.astype(np.float32)
        mask = mask / 255.0
        mask = np.clip(mask, 0, 1)
        return image, mask

    def _divide(self, image, mask, mean=None, std=None, target_size=None, resize_mode='inflexible', interpolation=None):
        if resize_mode.lower() == 'inflexible' and target_size:
            image, mask = self.__resize_inflexible_mode(image, mask, target_size, interpolation)
        elif resize_mode.lower() == 'flexible' and target_size:
            image, mask = self.__resize_flexible_mode(image, mask, target_size, interpolation)
        image = image.astype(np.float32)
        image = image / 255.0
        image = self.__get_standard_deviation(image, mean, std)
        image = np.clip(image, 0, 1)

        mask = mask.astype(np.float32)
        mask = mask / 255.0
        mask = np.clip(mask, 0, 1)
        return image, mask

    def _basic(self, image, mask, mean=None, std=None, target_size=None, resize_mode='inflexible', interpolation=None):
        if resize_mode.lower() == 'inflexible' and target_size:
            image, mask = self.__resize_inflexible_mode(image, mask, target_size, interpolation)
        elif resize_mode.lower() == 'flexible' and target_size:
            image, mask = self.__resize_flexible_mode(image, mask, target_size, interpolation)
        image = image.astype(np.uint8)
        image = self.__get_standard_deviation(image, mean, std)
        image = np.clip(image, 0, 255)
        mask = mask.astype(np.float32)
        mask = np.clip(mask, 0, 255)
        return image, mask

    def __call__(self, input, *args, **kargs):
        if self.mode == "divide":
            return self._divide(input, *args, **kargs)
        elif self.mode == "sub_divide":
            return self._sub_divide(input, *args, **kargs)
        else:
            return self._basic(input, *args, **kargs)