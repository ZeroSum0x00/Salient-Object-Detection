import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import metrics


class MAE(tf.keras.metrics.Metric):
    def __init__(self, 
                 name="MAE", **kwargs):
        super(MAE, self).__init__(**kwargs)
        self.mae = self.add_weight(name, initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        score = metrics.mean_absolute_error(y_true, y_pred)
        score = K.mean(score)
        self.mae.assign_add(score)
        
    def reset_state(self):
        self.mae.assign(0.)

    def result(self):
        return self.mae