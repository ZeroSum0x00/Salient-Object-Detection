import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy


class CascadeBCE(tf.keras.losses.Loss):
    def __init__(self, 
                 from_logits=False,
                 label_smoothing=0.0,
                 name="CascadeBCE", **kwargs):
        super(CascadeBCE, self).__init__(name=name, **kwargs)
        self.losses = BinaryCrossentropy(from_logits=from_logits, label_smoothing=label_smoothing)

    def __call__(self, y_true, y_pred):
        loss_list = []

        for y in y_pred:
            loss_list.append(self.losses(y_true, y))
        total_loss = tf.reduce_sum(loss_list)
        return total_loss