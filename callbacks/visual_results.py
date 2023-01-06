import os
import cv2
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.files import verify_folder


class VisualizerPerEpoch(tf.keras.callbacks.Callback):
    def __init__(self, val_dataset, n_samples=1, saved_path=None, show_frequency=5):
        self.image, self.mask = random.choice(val_dataset)
        self.n_samples = n_samples
        self.saved_path = saved_path
        self.show_frequency = show_frequency
        
    def infer(self):
        return self.model.architecture(self.image, training=False)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.show_frequency == 0 or epoch == self.show_frequency:
            d0, d1, d2, d3, d4, d5, d6 = self.infer()
            for index in range(self.n_samples):
                f, ax = plt.subplots(1, 9, figsize=(36, 4))
                ax[0].imshow(cv2.cvtColor((self.image[index] + 1) / 2, cv2.COLOR_BGR2RGB))
                ax[0].axis("off")
                ax[0].set_title("Image", fontsize=20)
                ax[1].imshow((self.mask[index] + 1) / 2)
                ax[1].axis("off")
                ax[1].set_title("Mask", fontsize=20)
                ax[2].imshow((d0[index] + 1) / 2)
                ax[2].axis("off")
                ax[2].set_title("Fused Mask", fontsize=20)
                ax[3].imshow((d1[index] + 1) / 2)
                ax[3].axis("off")
                ax[3].set_title("Side 1", fontsize=20)
                ax[4].imshow((d2[index] + 1) / 2)
                ax[4].axis("off")
                ax[4].set_title("Side 2", fontsize=20)
                ax[5].imshow((d3[index] + 1) / 2)
                ax[5].axis("off")
                ax[5].set_title("Side 3", fontsize=20)
                ax[6].imshow((d4[index] + 1) / 2)
                ax[6].axis("off")
                ax[6].set_title("Side 4", fontsize=20)
                ax[7].imshow((d5[index] + 1) / 2)
                ax[7].axis("off")
                ax[7].set_title("Side 5", fontsize=20)
                ax[8].imshow((d6[index] + 1) / 2)
                ax[8].axis("off")
                ax[8].set_title("Side 6", fontsize=20)
                if self.saved_path:
                    # saved_image_path = verify_folder(self.saved_path) + f'epoch_{epoch + 1}/'
                    saved_image_path = verify_folder(self.saved_path)
                    if not os.path.isdir(saved_image_path):
                        os.makedirs(saved_image_path)
                    # plt.savefig(saved_image_path + f'image_{index + 1}.png')
                    plt.savefig(saved_image_path + f'image{index + 1}-epoch{epoch}.png')
                    plt.close()
                else:
                    plt.show()