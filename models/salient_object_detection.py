import tensorflow as tf
from tensorflow.keras.models import load_model

from utils.logger import logger


class SOD(tf.keras.Model):
    def __init__(self, 
                 architecture, 
                 image_size=(288, 288, 3), 
                 is_training=False, 
                 **kwargs):
        super(SOD, self).__init__(**kwargs)
        self.architecture = architecture
        self.image_size = image_size
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        
    def compile(self, optimizer, loss, metrics=None, **kwargs):
        super(SOD, self).compile(**kwargs)
        self.optimizer = optimizer
        self.loss_object = loss
        self.list_metrics = metrics

    @property
    def metrics(self):
        if self.list_metrics:
            return [
                self.total_loss_tracker,
                *self.list_metrics,
            ]
        else:
            return [self.total_loss_tracker]

    def train_step(self, data):
        images, masks = data        
        with tf.GradientTape() as tape:
            model_result = self.architecture(images, training=True)
            y_pred       = model_result['saliency']
            loss_value   = self.architecture.calc_loss(y_true=masks, y_pred=y_pred, loss_object=self.loss_object)
        grads = tape.gradient(loss_value, self.architecture.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.architecture.trainable_variables))
        
        self.total_loss_tracker.update_state(loss_value)
        if self.list_metrics:
            for metric in self.list_metrics:
                metric.reset_state()
                metric.update_state(labels, y_pred)
        results = {m.name: m.result() for m in self.metrics}
        return results

    def test_step(self, data):
        images, masks = data
        model_result = self.architecture(images, training=False)
        y_pred       = model_result['saliency']
        loss_value   = self.architecture.calc_loss(y_true=masks, y_pred=y_pred, loss_object=self.loss_object)
        
        self.total_loss_tracker.update_state(loss_value)
        if self.list_metrics:
            for metric in self.list_metrics:
                metric.reset_state()
                metric.update_state(labels, y_pred)
        results = {m.name: m.result() for m in self.metrics}
        return results

    def call(self, inputs):
        try:
            pred = self.predict(inputs)
            return pred
        except:
            return inputs

    @tf.function
    def predict(self, inputs):
        pred = self.architecture.predict(inputs)
        return pred

    def save_weights(self, weight_path, save_format='tf', **kwargs):
        self.architecture.save_weights(weight_path, save_format=save_format, **kwargs)

    def load_weights(self, weight_objects):
        for weight in weight_objects:
            weight_path = weight['path']
            custom_objects = weight['custom_objects']
            if weight_path:
                self.architecture.build(input_shape=self.image_size)
                self.architecture.built = True
                self.architecture.load_weights(weight_path)
                logger.info("Load SOD weights from {}".format(weight_path))

    def save_models(self, weight_path, save_format='tf'):
        self.architecture.save(weight_path, save_format=save_format)

    def load_models(self, weight_objects):
        for weight in weight_objects:
            weight_path = weight['path']
            custom_objects = weight['custom_objects']
            if weight_path:
                self.architecture = load_model(weight_path, custom_objects=custom_objects)
                logger.info("Load SOD model from {}".format(weight_path))

    def get_config(self):
        config = super().get_config()
        config.update({
                "architecture": self.architecture,
                "image_size": self.image_size,
                "total_loss_tracker": self.total_loss_tracker,
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
