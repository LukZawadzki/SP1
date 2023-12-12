import keras.activations
import keras.layers as layers
from keras.applications.vgg16 import VGG16
import tensorflow as tf
from utils.gaussian import Gaussian_kernel
import numpy as np


def create_aspp():

    backbone = VGG16(False, 'imagenet', None, pooling=None)

    pooling_1 = backbone.get_layer("block3_pool")
    pooling_2 = backbone.get_layer("block4_pool")
    pooling_3 = backbone.get_layer("block5_pool")

    pooling_2.padding = "same"
    pooling_2.strides = (1, 1)

    pooling_3.padding = "same"
    pooling_3.strides = (1, 1)

    for layer in backbone.layers:
        if layer.name in ['block5_conv1', 'block5_conv2', 'block5_conv3']:
            layer.dilation_rate = (2, 2)
        layer.trainable = False

    concat = tf.concat([pooling_1.output, pooling_2.output, pooling_3.output], axis=-1)

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

    # aspp_pooling = layers.MaxPooling2D(2,1, padding="same")(concat)
    # aspp_dilation_1 = layers.Conv2D(256, 3,
    #                                    padding="same",
    #                                    activation="relu",
    #                                    dilation_rate=4,
    #                                    name="aspp_dilation_1")(aspp_pooling)
    # aspp_dilation_2 = layers.Conv2D(256, 3,
    #                                    padding="same",
    #                                    activation="relu",
    #                                    dilation_rate=8,
    #                                    name="aspp_dilation_2")(aspp_dilation_1)
    # aspp_dilation_3 = layers.Conv2D(256, 3,
    #                                    padding="same",
    #                                    activation="relu",
    #                                    dilation_rate=12,
    #                                    name="aspp_dilation_3")(aspp_dilation_2)
    #
    # aspp_conv_1 = layers.Conv2D(256, 1,
    #                                padding="valid",
    #                                activation="relu",
    #                                name="aspp_conv_1")(aspp_dilation_3)

    decoder_upsampling_1 = layers.UpSampling2D(name="decoder_upsampling_1")(aspp_output)

    decoder_conv_1 = layers.Conv2D(128, 3,
                                      padding="same",
                                      activation="relu",
                                      name="decoder_conv_1")(decoder_upsampling_1)

    decoder_upsampling_2 = layers.UpSampling2D(name="decoder_upsampling_2")(decoder_conv_1)

    decoder_conv_2 = layers.Conv2D(64, 3,
                                   padding="same",
                                   activation="relu",
                                   name="decoder_conv_2")(decoder_upsampling_2)

    decoder_upsampling_3 = layers.UpSampling2D(name="decoder_upsampling_3")(decoder_conv_2)

    decoder_conv_3 = layers.Conv2D(32, 3,
                                   padding="same",
                                   activation="relu",
                                   name="decoder_conv_3")(decoder_upsampling_3)

    # decoder_upsampling_4 = layers.UpSampling2D(name="decoder_upsampling_4")(decoder_conv_3)
    #
    # decoder_conv_4 = layers.Conv2D(16, 3,
    #                                padding="same",
    #                                activation="relu",
    #                                name="decoder_conv_4")(decoder_upsampling_4)

    # decoder_upsampling_5 = layers.UpSampling2D(name="decoder_upsampling_5")(decoder_conv_4)
    #
    # decoder_conv_5 = layers.Conv2D(8, 3,
    #                                padding="same",
    #                                activation="relu",
    #                                name="decoder_conv_5")(decoder_upsampling_5)
    #
    # decoder_upsampling_6 = layers.UpSampling2D(name="decoder_upsampling_6")(decoder_conv_5)
    #
    # decoder_conv_6 = layers.Conv2D(4, 3,
    #                                padding="same",
    #                                activation="relu",
    #                                name="decoder_conv_6")(decoder_upsampling_6)

    decoder_conv_7 = layers.Conv2D(1, 3,
                                      padding="same",
                                      name="decoder_conv_7")(decoder_conv_3)
    # normalization = layers.BatchNormalization()(decoder_conv_7)
    # activation = layers.Activation(keras.activations.hard_sigmoid, name='block10_out')(normalization)

    decoder_output = layers.Resizing(224, 224, name="decoder_output")(decoder_conv_7)

    # gaussian_kernel = tf.convert_to_tensor(Gaussian_kernel(l=10, sig=5))
    # gaussian_kernel = tf.expand_dims(gaussian_kernel, axis=-1)
    # gaussian_kernel = tf.expand_dims(gaussian_kernel, axis=-1)
    # gaussian = tf.nn.conv2d(decoder_output, gaussian_kernel, strides=[1, 1, 1, 1], padding='SAME', name='gaussian')
    # gaussian_resizing = layers.Resizing(28,28, name='gaussian_resizing')(gaussian)

    # center_bias_array = np.load("centerbias_mit1003.npy")
    # constant_bias = tf.constant(center_bias_array, name="const_bias")
    # constant_bias = tf.expand_dims(constant_bias, axis=-1)

    # center_bias = tf.keras.Input(tensor=tf.constant(tf.ones([256, 256, 1])), name='block10_in')
    # cb = layers.Resizing(28,28, name='block10_rs1')(center_bias)
    # cb_2 = layers.Flatten(name='block10_flat1')(cb)
    # p_cb = layers.Softmax(name='block10_soft1')(cb_2)
    # p_cb_2 = layers.Lambda(lambda x: tf.math.log(x), name='block10_lambda')(p_cb)
    # p_cb_3 = layers.Reshape((1, 28, 28, 1), name='block10_rs')(p_cb_2)
    # x10 = layers.Add(name='block10_add')([resizing, p_cb_3])

    # batch_normalization = layers.BatchNormalization(name='block10_bn')(gaussian_resizing)
    # activation = layers.Activation('sigmoid', name='block10_out')(batch_normalization)
    # output_resize = layers.Resizing(224, 224)(gaussian_resizing)
    # flatten = layers.Flatten()(decoder_conv_7)
    # softmax = layers.Softmax()(flatten)
    # final_reshape = layers.Reshape((224, 224, 1))(softmax)
    # softmax_output = layers.Resizing(224, 224)(final_reshape)

    sgd = tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9, nesterov=True)
    adam = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model = tf.keras.Model(inputs=backbone.input, outputs=decoder_output)
    model.compile(optimizer=adam,
                  loss="mse",
                  metrics=[tf.keras.metrics.KLD, "AUC", "accuracy"])
    return model
