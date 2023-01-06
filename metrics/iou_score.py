import tensorflow as tf
from tensorflow.keras import backend as K


class IOUScore(tf.keras.metrics.Metric):
    def __init__(self, 
                 smooth=1,
                 name="IOUScore", **kwargs):
        super(IOUScore, self).__init__(**kwargs)
        self.smooth = smooth
        self.iou = self.add_weight('iou', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        intersection = K.sum(K.abs(y_true * y_pred), axis=[0, 1, 2])
        union = K.sum(y_true, [0, 1, 2]) + K.sum(y_pred, [0, 1, 2]) - intersection
        iou_score = K.mean((intersection + self.smooth) / (union + self.smooth), axis=0)
        iou_score = tf.reduce_sum(iou_score)
        self.iou.assign_add(iou_score)
        
    def reset_state(self):
        self.iou.assign(0.)

    def result(self):
        return self.iou