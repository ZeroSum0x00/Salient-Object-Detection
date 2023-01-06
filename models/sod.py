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
        
    def compile(self, optimizer, loss, metrics, **kwargs):
        super(SOD, self).compile(**kwargs)
        self.optimizer = optimizer
        self.loss = loss
        self.list_metrics = metrics

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            *self.list_metrics,
        ]

    def train_step(self, data):
        for metric in self.list_metrics:
            metric.reset_state()
        images, masks = data        
        with tf.GradientTape() as tape:
            y_pred      = self.architecture(images, training=True)
            loss_value  = self.loss(y_true=masks, y_pred=y_pred)
        grads = tape.gradient(loss_value, self.architecture.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.architecture.trainable_variables))
        self.total_loss_tracker.update_state(loss_value)
        
        for metric in self.list_metrics:
            metric.update_state(masks, y_pred[0])
        results = {m.name: m.result() for m in self.metrics}
        return results

    def test_step(self, data):
        for metric in self.list_metrics:
            metric.reset_state()
        images, masks = data
        y_pred      = self.architecture(images, training=False)
        loss_value  = self.loss(y_true=masks, y_pred=y_pred)
        self.total_loss_tracker.update_state(loss_value)
        for metric in self.list_metrics:
            metric.update_state(masks, y_pred[0])
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
        d0, d1, d2, d3, d4, d5, d6 = self.architecture(inputs, training=False)
        return d0

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
                # "tar_loss_tracker": self.tar_loss_tracker
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)