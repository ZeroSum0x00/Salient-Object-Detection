import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger

from data_utils.data_flow import get_train_test_data
from utils.train_processing import create_folder_weights, train_prepare
from models import U2Net
from models import SOD
from losses import CascadeBCE
from metrics import FScore
from metrics import IOUScore
from callbacks import LossHistory
from callbacks import VisualizerPerEpoch

from configs import general_config as cfg


def train(data_path                   = cfg.DATA_PATH,
          data_anno_path              = cfg.DATA_ANNOTATION_PATH,
          data_dst_path               = cfg.DATA_DESTINATION_PATH,
          data_normalizer             = cfg.DATA_NORMALIZER,
          data_augmentation           = cfg.DATA_AUGMENTATION,
          check_data                  = cfg.CHECK_DATA,
          input_shape                 = cfg.TARGET_SIZE,
          load_memory                 = cfg.DATA_LOAD_MEMORY,
          batch_size                  = cfg.TRAIN_BATCH_SIZE,
          init_epoch                  = cfg.TRAIN_EPOCH_INIT,
          end_epoch                   = cfg.TRAIN_EPOCH_END,
          weight_type                 = cfg.TRAIN_WEIGHT_TYPE,
          weight_objects              = cfg.TRAIN_WEIGHT_OBJECTS,
          saved_path                  = cfg.TRAIN_SAVED_PATH,
          saved_weight_frequency      = cfg.TRAIN_SAVE_WEIGHT_FREQUENCY,
          show_frequency              = cfg.TRAIN_RESULT_SHOW_FREQUENCY,
          training_mode               = cfg.TRAIN_MODE):

    if train_prepare(training_mode):
        TRAINING_TIME_PATH = create_folder_weights(saved_path)

        train_generator, val_generator = get_train_test_data(data_path, 
                                                             data_dst_path, 
                                                             data_anno_path,
                                                             target_size=input_shape,
                                                             batch_size=batch_size,
                                                             augmentor=data_augmentation, 
                                                             normalizer=data_normalizer,
                                                             check_data=check_data,
                                                             load_memory=load_memory)
        num_classes = 1
        architecture = U2Net(filters=[16, 32, 64, 128, 256, 512], classes=num_classes)
        model = SOD(architecture=architecture, image_size=input_shape)

        if weight_type and weight_objects:
            if weight_type == "weights":
                model.load_weights(weight_objects)
            elif weight_type == "models":
                model.load_models(weight_objects)

        checkpoint = ModelCheckpoint(TRAINING_TIME_PATH + 'checkpoint_{epoch:04d}/saved_sod_weights', 
                                     monitor='val_f_score',
                                     verbose=1, 
                                     save_weights_only=True,
                                     save_freq="epoch")
        history = LossHistory(result_path=TRAINING_TIME_PATH)

        logger = CSVLogger(TRAINING_TIME_PATH + 'train_history.csv', separator=",", append=True)

        visualizer = VisualizerPerEpoch(val_generator, 
                                    batch_size, 
                                    saved_path=TRAINING_TIME_PATH + 'visualizer/', 
                                    show_frequency=show_frequency)
        
        callbacks = [checkpoint, logger, history, visualizer]

        loss = CascadeBCE()
        
        f_score = FScore(beta=0.3)
        # iou_score = IOUScore()
        
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        
        model.compile(optimizer=optimizer, loss=loss, metrics=[f_score])
        model.fit(train_generator,
                  steps_per_epoch     = train_generator.n // batch_size,
                  validation_data     = val_generator,
                  validation_steps    = val_generator.n // batch_size,
                  epochs              = end_epoch,
                  initial_epoch       = init_epoch,
                  callbacks           = callbacks)
        model.save_weights(TRAINING_TIME_PATH + 'best_weights', save_format="tf")

    
if __name__ == "__main__":
    train()