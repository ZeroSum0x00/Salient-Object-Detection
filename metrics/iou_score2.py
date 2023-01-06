import tensorflow as tf
from tensorflow.keras import backend as K
from metrics.metric_utils import expand_binary, gather_channels, round_if_needed, get_reduce_axes, average

SMOOTH = 1e-5

class IOUScore2(tf.keras.metrics.Metric):
    r""" The `Jaccard index`_, also known as Intersection over Union and the Jaccard similarity coefficient
    (originally coined coefficient de communauté by Paul Jaccard), is a statistic used for comparing the
    similarity and diversity of sample sets. The Jaccard coefficient measures similarity between finite sample sets,
    and is defined as the size of the intersection divided by the size of the union of the sample sets:

    .. math:: J(A, B) = \frac{A \cap B}{A \cup B}

    Args:
        gt: ground truth 4D keras tensor(B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor(B, H, W, C) or (B, C, H, W)
        class_weights: 1. or list of class weights, len(weights) = C
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch(B),
            else over whole batch
        threshold: value to round predictions(use ``>`` comparison), if ``None`` prediction will not be round

    Returns:
        IoU / Jaccard score in range[0, 1]

    .. _`Jaccard index`: https: // en.wikipedia.org / wiki / Jaccard_index

    """
    def __init__(self, 
                 class_weights=1, 
                 class_indexes=None, 
                 smooth=SMOOTH, 
                 per_image=False, 
                 threshold=None,
                 name="IOUScore2", **kwargs):
        super(IOUScore2, self).__init__(**kwargs)
        self.class_weights = class_weights
        self.class_indexes = class_indexes
        self.smooth = smooth
        self.per_image = per_image
        self.threshold = threshold        
        self.iou_score = self.add_weight(name, initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        if y_true.shape[-1] == 1:
            y_true, y_pred = expand_binary(y_true), expand_binary(y_pred)
            
        y_true, y_pred = gather_channels(y_true, y_pred, indexes=self.class_indexes)
        y_pred = round_if_needed(y_pred, self.threshold)
        axes = get_reduce_axes(self.per_image)

        intersection = K.sum(gt * pr, axis=axes)
        union = K.sum(gt + pr, axis=axes) - intersection

        score = (intersection + self.smooth) / (union + self.smooth)
        score = average(score, self.per_image, self.class_weights)
        self.iou_score.assign_add(score)

    def reset_state(self):
        self.iou_score.assign(0.)

    def result(self):
        return self.iou_score