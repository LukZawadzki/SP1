import keras.layers as layers
import tensorflow as tf

from config import INPUT_IMAGE_SIZE


def create_encoder_decoder():
    backbone = tf.keras.applications.VGG19(False, 'imagenet', input_shape=(*INPUT_IMAGE_SIZE, 3))

    for layer in backbone.layers:
        layer.trainable = False

    decoder_upsampling_1 = layers.UpSampling2D(name='decoder_upsampling_1')(backbone.output)
    decoder_conv_1_1 = layers.Conv2D(256, 3, padding='same', activation='relu')(decoder_upsampling_1)
    decoder_conv_1_2 = layers.Conv2D(256, 3, padding='same', activation='relu')(decoder_conv_1_1)

    decoder_upsampling_2 = layers.UpSampling2D(name='decoder_upsampling_2')(decoder_conv_1_2)
    decoder_conv_2_1 = layers.Conv2D(128, 3, padding='same', activation='relu')(decoder_upsampling_2)
    decoder_conv_2_2 = layers.Conv2D(128, 3, padding='same', activation='relu')(decoder_conv_2_1)

    decoder_upsampling_3 = layers.UpSampling2D(name='decoder_upsampling_3')(decoder_conv_2_2)
    decoder_conv_3_1 = layers.Conv2D(64, 3, padding='same', activation='relu')(decoder_upsampling_3)
    decoder_conv_3_2 = layers.Conv2D(64, 3, padding='same', activation='relu')(decoder_conv_3_1)
    decoder_conv_3_3 = layers.Conv2D(64, 3, padding='same', activation='relu')(decoder_conv_3_2)

    decoder_upsampling_4 = layers.UpSampling2D(name='decoder_upsampling_4')(decoder_conv_3_3)
    decoder_conv_4_1 = layers.Conv2D(32, 3, padding='same', activation='relu')(decoder_upsampling_4)
    decoder_conv_4_2 = layers.Conv2D(32, 3, padding='same', activation='relu')(decoder_conv_4_1)
    decoder_conv_4_3 = layers.Conv2D(32, 3, padding='same', activation='relu')(decoder_conv_4_2)

    decoder_upsampling_5 = layers.UpSampling2D(name='decoder_upsampling_5')(decoder_conv_4_3)
    decoder_conv_5_1 = layers.Conv2D(16, 3, padding='same', activation='relu')(decoder_upsampling_5)
    decoder_conv_5_2 = layers.Conv2D(16, 3, padding='same', activation='relu')(decoder_conv_5_1)
    decoder_conv_5_3 = layers.Conv2D(1, 3, padding='same', activation='relu')(decoder_conv_5_2)

    # sgd = tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9, nesterov=True)
    adam = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model = tf.keras.Model(inputs=backbone.input, outputs=decoder_conv_5_3)

    model.compile(optimizer=adam, loss="mse", metrics=[tf.keras.metrics.KLD, "AUC", "accuracy"])
    # model.summary()
    return model
