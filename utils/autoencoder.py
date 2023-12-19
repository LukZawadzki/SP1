import keras.initializers.initializers_v2
import keras.layers as layers
from keras.applications.vgg16 import VGG16
import tensorflow as tf
from config import IMAGE_SIZE


def create_autoencoder():
    backbone = keras.applications.VGG19(False, 'imagenet', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    for layer in backbone.layers:
        layer.trainable = False

    decoder_upsampling_1 = layers.UpSampling2D(name='decoder_upsampling_1')(backbone.output)
    decoder_conv_1_1 = layers.Conv2D(256, 5, padding='same', activation='relu')(decoder_upsampling_1)
    decoder_conv_1_2 = layers.Conv2D(256, 5, padding='same', activation='relu')(decoder_conv_1_1)

    decoder_upsampling_2 = layers.UpSampling2D(name='decoder_upsampling_2')(decoder_conv_1_2)
    decoder_conv_2_1 = layers.Conv2D(128, 3, padding='same', activation='relu')(decoder_upsampling_2)
    decoder_conv_2_2 = layers.Conv2D(128, 3, padding='same', activation='relu')(decoder_conv_2_1)

    decoder_upsampling_3 = layers.UpSampling2D(name='decoder_upsampling_3')(decoder_conv_2_2)
    decoder_conv_3_1 = layers.Conv2D(64, 3, padding='same', activation='relu')(decoder_upsampling_3)
    decoder_conv_3_2 = layers.Conv2D(64, 3, padding='same', activation='relu')(decoder_conv_3_1)

    decoder_upsampling_4 = layers.UpSampling2D(name='decoder_upsampling_4')(decoder_conv_3_2)
    decoder_conv_4_1 = layers.Conv2D(32, 3, padding='same', activation='relu')(decoder_upsampling_4)
    decoder_conv_4_2 = layers.Conv2D(32, 3, padding='same', activation='relu')(decoder_conv_4_1)

    decoder_upsampling_5 = layers.UpSampling2D(name='decoder_upsampling_5')(decoder_conv_4_2)
    decoder_conv_5_1 = layers.Conv2D(16, 3, padding='same', activation='relu')(decoder_upsampling_5)
    decoder_conv_5_2 = layers.Conv2D(1, 3, padding='same', activation='relu')(decoder_conv_5_1)

    # sgd = tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9, nesterov=True)
    sgd = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
    adam = tf.keras.optimizers.Adam(learning_rate=5e-4)
    model = tf.keras.Model(inputs=backbone.input, outputs=decoder_conv_5_2)
    model.compile(optimizer=adam, loss=tf.losses.huber, metrics=[tf.keras.metrics.KLD, "AUC", "accuracy"])
    model.summary()
    return model
