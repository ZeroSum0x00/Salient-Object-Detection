import tensorflow as tf
from tensorflow.keras import backend as K
from metrics.metric_utils import expand_binary, gather_channels, round_if_needed, get_reduce_axes, average

SMOOTH = 1e-5

class FScore(tf.keras.metrics.Metric):
    r"""The F-score (Dice coefficient) can be interpreted as a weighted average of the precision and recall,
    where an F-score reaches its best value at 1 and worst score at 0.
    The relative contribution of ``precision`` and ``recall`` to the F1-score are equal.
    The formula for the F score is:

    .. math:: 
        F_beta(precision, recall) = ((1 + beta**2) * (precision * recall)) / (beta**2 * precision + recall)

    The formula in terms of *Type I* and *Type II* errors:

    .. math:: F_\beta(A, B) = \frac{(1 + \beta^2) TP} {(1 + \beta^2) TP + \beta^2 FN + FP}


    where:
        TP - true positive;
        FP - false positive;
        FN - false negative;

    Args:
        y_true: ground truth 4D keras tensor (B, H, W, C)
        y_pred: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights, len(weights) = C
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        beta: f-score coefficient
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B), else over whole batch
        threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round

    Returns:
        F-score in range [0, 1]

    """
    def __init__(self, 
                 beta,
                 class_weights=1, 
                 class_indexes=None, 
                 smooth=SMOOTH, 
                 per_image=False, 
                 threshold=None,
                 name="FScore", **kwargs):
        super(FScore, self).__init__(**kwargs)
        self.beta = beta
        self.class_weights = class_weights
        self.class_indexes = class_indexes
        self.smooth = smooth
        self.per_image = per_image
        self.threshold = threshold        
        self.fscore = self.add_weight(name, initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if y_true.shape[-1] == 1:
            y_true, y_pred = expand_binary(y_true), expand_binary(y_pred)
            
        y_true, y_pred = gather_channels(y_true, y_pred, indexes=self.class_indexes)
        y_pred = round_if_needed(y_pred, self.threshold)
        axes = get_reduce_axes(self.per_image)
        tp = K.cast(K.sum(y_true * y_pred, axis=axes), dtype=tf.float32)
        fp = K.cast(K.sum(y_pred, axis=axes), dtype=tf.float32) - tp
        fn = K.cast(K.sum(y_true, axis=axes), dtype=tf.float32) - tp
        
        score = ((1 + self.beta ** 2) * tp + self.smooth) / ((1 + self.beta ** 2) * tp + self.beta ** 2 * fn + fp + self.smooth)
        score= average(score, self.per_image, self.class_weights)
        self.fscore.assign_add(score)

    def reset_state(self):
        self.fscore.assign(0.)

    def result(self):
        return self.fscore
