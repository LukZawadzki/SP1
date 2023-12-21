import keras.layers as layers
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from config import IMAGE_SIZE


def kld(y_true, y_pred, eps=1e-7):
    """This function computes the Kullback-Leibler divergence between ground
       truth saliency maps and their predictions. Values are first divided by
       their sum for each image to yield a distribution that adds to 1.

    Args:
        y_true (tensor, float32): A 4d tensor that holds the ground truth
                                  saliency maps with values between 0 and 255.
        y_pred (tensor, float32): A 4d tensor that holds the predicted saliency
                                  maps with values between 0 and 1.
        eps (scalar, float, optional): A small factor to avoid numerical
                                       instabilities. Defaults to 1e-7.

    Returns:
        tensor, float32: A 0D tensor that holds the averaged error.
    """

    sum_per_image = tf.reduce_sum(y_true, axis=(1, 2, 3), keepdims=True)
    y_true /= eps + sum_per_image

    sum_per_image = tf.reduce_sum(y_pred, axis=(1, 2, 3), keepdims=True)
    y_pred /= eps + sum_per_image
    loss = y_true * tf.math.log(eps + y_true / (eps + y_pred))
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=(1, 2, 3)))

    return loss


class SaliencyNormalizationLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SaliencyNormalizationLayer, self).__init__(**kwargs)

    def call(self, maps, **kwargs):
        # Assuming inputs is a saliency map (3D tensor: batch_size x height x width)
        min_per_image = tf.reduce_min(maps, axis=(1, 2, 3), keepdims=True)
        maps -= min_per_image
        max_per_image = tf.reduce_max(maps, axis=(1, 2, 3), keepdims=True)
        return tf.divide(maps, tf.keras.backend.epsilon() + max_per_image, name="output")


def create_aspp():
    backbone = VGG16(False, 'imagenet', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), pooling=None)

    new_backbone = [layers.InputLayer((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))]
    pooling_layers = []

    for i in range(1, len(backbone.layers)):
        previous_layer = new_backbone[i - 1]
        if isinstance(previous_layer, layers.InputLayer):
            output = previous_layer.output
        else:
            output = previous_layer
        if "pool" in backbone.get_layer(index=i).name:
            if i >= 13:
                new_layer = layers.MaxPooling2D(pool_size=2, strides=1, padding="same")(output)
            else:
                new_layer = layers.MaxPooling2D()(output)
            pooling_layers.append(new_layer)
        else:
            old_layer = backbone.get_layer(index=i)
            new_layer = layers.Conv2D(old_layer.filters, old_layer.kernel_size, old_layer.strides,
                                      old_layer.padding, dilation_rate=old_layer.dilation_rate,
                                      weights=old_layer.get_weights())(output)

        new_backbone.append(new_layer)

    concat = tf.concat([pooling_layers[2], pooling_layers[3], pooling_layers[4]], axis=-1)

    aspp_conv_1 = layers.Conv2D(256, 1,
                                padding="same",
                                activation="relu",
                                name="aspp_conv_1")(concat)
    aspp_dilation_1 = layers.Conv2D(256, 3,
                                    padding="same",
                                    activation="relu",
                                    dilation_rate=4,
                                    name="aspp_dilation_1")(concat)
    aspp_dilation_2 = layers.Conv2D(256, 3,
                                    padding="same",
                                    activation="relu",
                                    dilation_rate=8,
                                    name="aspp_dilation_2")(concat)
    aspp_dilation_3 = layers.Conv2D(256, 3,
                                    padding="same",
                                    activation="relu",
                                    dilation_rate=12,
                                    name="aspp_dilation_3")(concat)

    aspp_conv_2 = layers.Conv2D(256, 1,
                                padding="valid",
                                activation="relu",
                                name="aspp_conv_2")(concat)

    aspp_concat = tf.concat([aspp_conv_1, aspp_dilation_1, aspp_dilation_2, aspp_dilation_3, aspp_conv_2],
                            axis=-1, name="aspp_concat")

    aspp_output = layers.Conv2D(256, 1,
                                padding="same",
                                activation="relu",
                                name="aspp_output")(aspp_concat)

    decoder_upsampling_1 = layers.UpSampling2D(name="decoder_upsampling_1", interpolation="bilinear")(aspp_output)

    decoder_conv_1 = layers.Conv2D(128, 3,
                                   padding="same",
                                   activation="relu",
                                   name="decoder_conv_1")(decoder_upsampling_1)

    decoder_upsampling_2 = layers.UpSampling2D(name="decoder_upsampling_2", interpolation="bilinear")(decoder_conv_1)

    decoder_conv_2 = layers.Conv2D(64, 3,
                                   padding="same",
                                   activation="relu",
                                   name="decoder_conv_2")(decoder_upsampling_2)

    decoder_upsampling_3 = layers.UpSampling2D(name="decoder_upsampling_3", interpolation="bilinear")(decoder_conv_2)

    decoder_conv_3 = layers.Conv2D(32, 3,
                                   padding="same",
                                   activation="relu",
                                   name="decoder_conv_3")(decoder_upsampling_3)

    decoder_conv_7 = layers.Conv2D(1, 3, padding="same", name="decoder_conv_7")(decoder_conv_3)
    normalization = SaliencyNormalizationLayer(name="SaliencyNormalizationLayer")(decoder_conv_7)

    sgd = tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9, nesterov=True)
    adam = tf.keras.optimizers.Adam(learning_rate=1e-5)

    test_model = tf.keras.Model(inputs=new_backbone[0].input, outputs=normalization)
    for layer in test_model.layers:
        if '2d' in layer.name:
            layer.trainable = False
    test_model.compile(optimizer=adam,
                       loss=kld,
                       metrics=[tf.keras.metrics.KLD, "AUC", "accuracy", "mse"])
    # print(test_model.summary())
    return test_model
