import os
import cv2


class ParseAnnotations:
    def __init__(self, 
                 data_dir, 
                 annotation_dir,
                 load_memory, 
                 check_data):
        self.data_dir          = data_dir
        self.annotation_dir    = annotation_dir if annotation_dir else data_dir
        self.load_memory       = load_memory
        self.check_data        = check_data

    def __call__(self, anno_files):
        data_extraction = []
        for anno_file in anno_files:
            anno_path = os.path.join(self.annotation_dir, anno_file)
            if self.check_data:
                image_file = anno_file.replace('png', 'jpg')
                image_path = os.path.join(self.data_dir, image_file)
                try:
                    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                    if len(img.shape) != 3:
                        print(f"Error: Image file {image_file} must be 3 channel in shape")
                        continue
                    try:
                        mask = cv2.imread(anno_path, cv2.IMREAD_UNCHANGED)
                        if len(mask.shape) != 2:
                            print(f"Error: Mask file {anno_file} must be 2 channel in shape")
                            continue
                    except:
                        print(f"Error: Mask file {anno_file} is missing or not in correct format")
                except Exception as e:
                    print(f"Error: File {image_file} is can't loaded: {e}")
                    continue
                    
            info_dict = {}
            info_dict['image_name'] = anno_file.replace('png', 'jpg')
            info_dict['mask_name'] = anno_file
            info_dict['image'] = None
            info_dict['mask'] = None
            info_dict['shape'] = None


            if self.load_memory:
                img = cv2.imread(os.path.join(self.data_dir, anno_file.replace('xml', 'jpg')))
                height, width, _ = img.shape
                info_dict['shape'] = (height, width)
                info_dict['image'] = img
                mask = cv2.imread(anno_path, cv2.IMREAD_UNCHANGED)
                info_dict['mask'] = mask

            data_extraction.append(info_dict)
        return data_extraction