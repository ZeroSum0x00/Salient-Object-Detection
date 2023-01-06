import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import metrics


class BinaryAccuracy(tf.keras.metrics.Metric):
    def __init__(self, 
                 name="BinaryAccuracy", **kwargs):
        super(BinaryAccuracy, self).__init__(**kwargs)
        self.accuracy = self.add_weight(name, initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        score = K.mean(metrics.binary_accuracy(y_true, y_pred))
        self.accuracy.assign_add(score)
        
    def reset_state(self):
        self.accuracy.assign(0.)

    def result(self):
        return self.accuracy