from tensorflow.keras import backend as K


def expand_binary(x):
    # remove last dim
    x = K.squeeze(x, axis=-1)
    # scale to 0 or 1
    x = K.round(x)
    x = K.cast(x, 'int32')
    x = K.one_hot(x, 2)
    return x


def _gather_channels(x, indexes):
    """Slice tensor along channels axis by given indexes"""
    x = K.permute_dimensions(x, (3, 0, 1, 2))
    x = K.gather(x, indexes)
    x = K.permute_dimensions(x, (1, 2, 3, 0))
    return x


def gather_channels(*xs, indexes=None):
    """Slice tensors along channels axis by given indexes"""
    if indexes is None:
        return xs
    elif isinstance(indexes, (int)):
        indexes = [indexes]
    xs = [_gather_channels(x, indexes=indexes) for x in xs]
    return xs


def round_if_needed(x, threshold):
    if threshold is not None:
        x = K.greater(x, threshold)
        x = K.cast(x, K.floatx())
    return x


def get_reduce_axes(per_image):
    axes = [1, 2]
    if not per_image:
        axes.insert(0, 0)
    return axes


def average(x, per_image=False, class_weights=None):
    if per_image:
        x = K.mean(x, axis=0)
    if class_weights is not None:
        x = x * class_weights
    return K.mean(x)



# ######################################
# from tensorflow.keras import backend as K

# SMOOTH = 1e-5

# def _f_score(y_true, y_pred, beta=1, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None):
#     r"""The F-score (Dice coefficient) can be interpreted as a weighted average of the precision and recall,
#     where an F-score reaches its best value at 1 and worst score at 0.
#     The relative contribution of ``precision`` and ``recall`` to the F1-score are equal.
#     The formula for the F score is:

#     .. math:: F_\beta(precision, recall) = (1 + \beta^2) \frac{precision \cdot recall}
#         {\beta^2 \cdot precision + recall}

#     The formula in terms of *Type I* and *Type II* errors:

#     .. math:: F_\beta(A, B) = \frac{(1 + \beta^2) TP} {(1 + \beta^2) TP + \beta^2 FN + FP}


#     where:
#         TP - true positive;
#         FP - false positive;
#         FN - false negative;

#     Args:
#         y_true: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
#         y_pred: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
#         class_weights: 1. or list of class weights, len(weights) = C
#         class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
#         beta: f-score coefficient
#         smooth: value to avoid division by zero
#         per_image: if ``True``, metric is calculated as mean over images in batch (B),
#             else over whole batch
#         threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round

#     Returns:
#         F-score in range [0, 1]

#     """
#     if y_true.shape[-1] == 1:
#         # assuming binary
#         y_true, y_pred = expand_binary(y_true), expand_binary(y_pred)

#     y_true, y_pred = gather_channels(y_true, y_pred, indexes=class_indexes)
#     y_pred = round_if_needed(y_pred, threshold)
#     axes = get_reduce_axes(per_image)

#     # calculate score
#     tp = K.cast(K.sum(y_true * y_pred, axis=axes), "float64")
#     fp = K.cast(K.sum(y_pred, axis=axes), 'float64') - tp
#     fn = K.cast(K.sum(y_true, axis=axes), 'float64') - tp

#     score = ((1 + beta ** 2) * tp + smooth) \
#           / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
#     score = average(score, per_image, class_weights)

#     return score


# def f1_score(class_weights=1):
#     def f1_score(y_true, y_pred):
#         return _f_score(y_true, y_pred, beta=1, class_weights=class_weights)
#     return f1_score