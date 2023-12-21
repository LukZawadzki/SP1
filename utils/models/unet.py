from keras import layers
import keras
import tensorflow as tf


def create_unet():
    backbone = keras.applications.VGG19(False, 'imagenet', input_shape=(224, 224, 3))

    for layer in backbone.layers:
        layer.trainable = False

    block1_conv2 = backbone.get_layer('block1_conv2')  # 224 x 224
    block2_conv2 = backbone.get_layer('block2_conv2')  # 112 x 112
    block3_conv4 = backbone.get_layer('block3_conv4')  # 56 x 56
    block4_conv4 = backbone.get_layer('block4_conv4')  # 28 x 28
    block5_conv4 = backbone.get_layer('block5_conv4')  # 14 x 14

    decoder_upsampling_1 = layers.UpSampling2D(name='decoder_upsampling_1')(backbone.output)
    concat_1 = layers.concatenate([decoder_upsampling_1, block5_conv4.output])
    decoder_conv_1_1 = layers.Conv2DTranspose(64, 7, padding='same', activation='relu')(concat_1)
    decoder_conv_1_2 = layers.Conv2DTranspose(64, 7, padding='same', activation='relu')(decoder_conv_1_1)
    decoder_conv_1_3 = layers.Conv2DTranspose(64, 7, padding='same', activation='relu')(decoder_conv_1_2)

    decoder_upsampling_2 = layers.UpSampling2D(name='decoder_upsampling_2')(decoder_conv_1_3)
    concat_2 = layers.concatenate([decoder_upsampling_2, block4_conv4.output])
    decoder_conv_2_1 = layers.Conv2DTranspose(128, 3, padding='same', activation='relu')(concat_2)
    decoder_conv_2_2 = layers.Conv2DTranspose(128, 3, padding='same', activation='relu')(decoder_conv_2_1)
    decoder_conv_2_3 = layers.Conv2DTranspose(128, 3, padding='same', activation='relu')(decoder_conv_2_2)

    decoder_upsampling_3 = layers.UpSampling2D(name='decoder_upsampling_3')(decoder_conv_2_3)
    concat_3 = layers.concatenate([decoder_upsampling_3, block3_conv4.output])
    decoder_conv_3_1 = layers.Conv2DTranspose(256, 3, padding='same', activation='relu')(concat_3)
    decoder_conv_3_2 = layers.Conv2DTranspose(256, 3, padding='same', activation='relu')(decoder_conv_3_1)
    decoder_conv_3_3 = layers.Conv2DTranspose(256, 3, padding='same', activation='relu')(decoder_conv_3_2)

    decoder_upsampling_4 = layers.UpSampling2D(name='decoder_upsampling_4')(decoder_conv_3_3)
    concat_4 = layers.concatenate([decoder_upsampling_4, block2_conv2.output])
    decoder_conv_4_1 = layers.Conv2DTranspose(256, 3, padding='same', activation='relu')(concat_4)
    decoder_conv_4_2 = layers.Conv2DTranspose(256, 3, padding='same', activation='relu')(decoder_conv_4_1)
    decoder_conv_4_3 = layers.Conv2DTranspose(256, 3, padding='same', activation='relu')(decoder_conv_4_2)

    decoder_upsampling_5 = layers.UpSampling2D(name='decoder_upsampling_5')(decoder_conv_4_3)
    concat_5 = layers.concatenate([decoder_upsampling_5, block1_conv2.output])
    decoder_conv_5_1 = layers.Conv2DTranspose(512, 3, padding='same', activation='relu')(concat_5)
    decoder_conv_5_2 = layers.Conv2DTranspose(512, 3, padding='same', activation='relu')(decoder_conv_5_1)
    decoder_conv_5_4 = layers.Conv2DTranspose(1, 3, padding='same', activation='relu')(decoder_conv_5_2)

    sgd = tf.keras.optimizers.SGD(learning_rate=1e-3, decay=0.0005, momentum=0.9, nesterov=True)
    # adam = tf.keras.optimizers.Adam(learning_rate=2e-5)
    model = tf.keras.Model(inputs=backbone.input, outputs=decoder_conv_5_4)
    model.compile(optimizer=sgd, loss="mse", metrics=[tf.keras.metrics.KLD, "AUC", "accuracy", "mse"])
    model.summary()
    return model
