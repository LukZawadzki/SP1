import tensorflow as tf


class SaliencyNormalizationLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SaliencyNormalizationLayer, self).__init__(**kwargs)

    def call(self, maps, **kwargs):
        # Assuming inputs is a saliency map (3D tensor: batch_size x height x width)
        min_per_image = tf.reduce_min(maps, axis=(1, 2, 3), keepdims=True)
        maps -= min_per_image
        max_per_image = tf.reduce_max(maps, axis=(1, 2, 3), keepdims=True)
        return tf.divide(maps, tf.keras.backend.epsilon() + max_per_image, name="output")
