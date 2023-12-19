import keras.layers as layers
from keras.applications.vgg16 import VGG16
import tensorflow as tf


def create_aspp():

    backbone = VGG16(False, 'imagenet', input_shape=(224, 224, 3), pooling=None)

    pooling_1 = backbone.get_layer("block3_pool")
    pooling_2 = backbone.get_layer("block4_pool")
    pooling_3 = backbone.get_layer("block5_pool")

    new_backbone = [layers.InputLayer((120, 160, 3))]
    pooling_layers = []

    for i in range(1, len(backbone.layers)):
        previous_layer = new_backbone[i-1]
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

    # for layer in backbone.layers:
    #     if layer.name in ['block5_conv1', 'block5_conv2', 'block5_conv3']:
    #         layer.dilation_rate = (2, 2)
    #     layer.trainable = False

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
    # model = tf.keras.Model(inputs=backbone.input, outputs=decoder_output)
    # model.compile(optimizer=adam,
    #               loss="mse",
    #               metrics=[tf.keras.metrics.KLD, "AUC", "accuracy"])
    # print(model.summary())

    test_model = tf.keras.Model(inputs=new_backbone[0].input, outputs=decoder_conv_7)
    for layer in test_model.layers:
        if '2d' in layer.name:
            layer.trainable = False
    test_model.compile(optimizer=adam,
                       loss="mse",
                       metrics=[tf.keras.metrics.KLD, "AUC", "accuracy", "mse"])
    # print(test_model.summary())
    return test_model
