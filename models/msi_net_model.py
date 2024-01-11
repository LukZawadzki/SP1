import keras.layers as layers
import tensorflow as tf
from keras.applications.vgg16 import VGG16

from config import INPUT_IMAGE_SIZE
from utils.layers import SaliencyNormalizationLayer
from utils.metrics import kld


def create_msi_net():
    backbone = VGG16(False, 'imagenet', input_shape=(*INPUT_IMAGE_SIZE, 3), pooling=None)

    new_backbone = [layers.InputLayer((*INPUT_IMAGE_SIZE, 3))]
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
            new_layer = layers.Conv2D(
                old_layer.filters,
                old_layer.kernel_size,
                old_layer.strides,
                old_layer.padding,
                dilation_rate=old_layer.dilation_rate,
                weights=old_layer.get_weights()
            )(output)

        new_backbone.append(new_layer)

    concat = tf.concat([pooling_layers[2], pooling_layers[3], pooling_layers[4]], axis=-1)

    aspp_conv_1 = layers.Conv2D(
        256,
        1,
        padding="same",
        activation="relu",
        name="aspp_conv_1"
    )(concat)
    aspp_dilation_1 = layers.Conv2D(
        256,
        3,
        padding="same",
        activation="relu",
        dilation_rate=4,
        name="aspp_dilation_1"
    )(concat)
    aspp_dilation_2 = layers.Conv2D(
        256,
        3,
        padding="same",
        activation="relu",
        dilation_rate=8,
        name="aspp_dilation_2"
    )(concat)
    aspp_dilation_3 = layers.Conv2D(
        256,
        3,
        padding="same",
        activation="relu",
        dilation_rate=12,
        name="aspp_dilation_3"
    )(concat)

    aspp_conv_2 = layers.Conv2D(
        256,
        1,
        padding="valid",
        activation="relu",
        name="aspp_conv_2"
    )(concat)

    aspp_concat = tf.concat(
        [aspp_conv_1, aspp_dilation_1, aspp_dilation_2, aspp_dilation_3, aspp_conv_2],
        axis=-1,
        name="aspp_concat"
    )

    aspp_output = layers.Conv2D(
        256,
        1,
        padding="same",
        activation="relu",
        name="aspp_output"
    )(aspp_concat)

    decoder_upsampling_1 = layers.UpSampling2D(name="decoder_upsampling_1", interpolation="bilinear")(aspp_output)

    decoder_conv_1 = layers.Conv2D(
        128,
        3,
        padding="same",
        activation="relu",
        name="decoder_conv_1"
    )(decoder_upsampling_1)

    decoder_upsampling_2 = layers.UpSampling2D(name="decoder_upsampling_2", interpolation="bilinear")(decoder_conv_1)

    decoder_conv_2 = layers.Conv2D(
        64,
        3,
        padding="same",
        activation="relu",
        name="decoder_conv_2"
    )(decoder_upsampling_2)

    decoder_upsampling_3 = layers.UpSampling2D(name="decoder_upsampling_3", interpolation="bilinear")(decoder_conv_2)

    decoder_conv_3 = layers.Conv2D(
        32,
        3,
        padding="same",
        activation="relu",
        name="decoder_conv_3"
    )(decoder_upsampling_3)

    decoder_conv_7 = layers.Conv2D(1, 3, padding="same", name="decoder_conv_7")(decoder_conv_3)
    normalization = SaliencyNormalizationLayer(name="SaliencyNormalizationLayer")(decoder_conv_7)

    # sgd = tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9, nesterov=True)
    adam = tf.keras.optimizers.Adam(learning_rate=1e-5)

    model = tf.keras.Model(inputs=new_backbone[0].input, outputs=normalization)
    for layer in model.layers:
        if '2d' in layer.name:
            layer.trainable = False

    model.compile(
        optimizer=adam,
        loss=kld,
        metrics=[tf.keras.metrics.KLD, "AUC", "accuracy", "mse"]
    )
    # test_model.summary()
    return model
