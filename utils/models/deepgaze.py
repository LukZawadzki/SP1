import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import Resizing


def gaussian_kernel(l=5, sig=1):
    """\
    creates gaussian kernel with side length l and a sigma of sig
    https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel / np.sum(kernel)


def create_deepgaze():
    vgg16 = tf.keras.applications.vgg16.VGG16(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=(256, 256, 3),
        pooling=True
    )

    names = (
        'block4_conv1',
        'block4_conv2',
        'block4_conv3',
        'block4_pool',
        'block5_conv1',
        'block5_conv2',
        'block5_conv3',
        'block5_pool'
    )

    for layer in vgg16.layers:
        layer.trainable = layer.name in names

    x_conv4_3 = vgg16.get_layer('block4_conv3').output  # 28*28*512
    x_conv4_pool = vgg16.get_layer('block4_pool').output  # 14*14*512
    x_conv5_1 = vgg16.get_layer('block5_conv1').output  # 14*14*512
    x_conv5_2 = vgg16.get_layer('block5_conv2').output  # 14*14*512
    x_conv5_3 = vgg16.get_layer('block5_conv3').output  # 14*14*512

    # resize all 5 layers into 128x128

    feature_pyramid = [x_conv4_3, x_conv4_pool, x_conv5_1, x_conv5_2, x_conv5_3]

    x6 = [Resizing(128, 128)(x) for x in feature_pyramid]  # 128 x 128 x 512*3

    x7 = layers.Concatenate(name='block7_cat1')(x6)

    # Readout Network
    l2 = tf.keras.regularizers.L2(1e-2)
    x8_1 = tf.keras.layers.Conv2D(16, 1, activation='relu', kernel_regularizer=l2, name='block8_conv1')(x7)  # 128*128*16
    x8_2 = tf.keras.layers.Conv2D(32, 1, activation='relu', kernel_regularizer=l2, name='block8_conv2')(x8_1)  # 128*128*32
    x8_3 = tf.keras.layers.Conv2D(2, 1, activation='relu', kernel_regularizer=l2, name='block8_conv3')(x8_2)  # 128*128*2
    x8_4 = tf.keras.layers.Conv2D(1, 1, activation='relu', kernel_regularizer=l2, name='block8_conv4')(x8_3)  # 128*128*1
    x8_5 = Resizing(256, 256, name='block8_rs')(x8_4)

    # Gaussian Smoothing
    kernel = tf.convert_to_tensor(gaussian_kernel(l=10, sig=5))
    kernel = tf.expand_dims(kernel, axis=-1)
    kernel = tf.expand_dims(kernel, axis=-1)
    x9 = tf.nn.conv2d(x8_5, kernel, strides=[1, 1, 1, 1], padding='same', name='block9_conv1')

    x9_1 = Resizing(28, 28, name='block9_rs1')(x9)

    x = layers.BatchNormalization(name='block10_bn')(x9_1)
    x = layers.Activation('sigmoid', name='block10_out')(x)

    outputs = Resizing(120, 160)(x)

    model = tf.keras.Model(vgg16.input, outputs)

    sgd = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)

    model.compile(
        optimizer=sgd,
        loss=tf.keras.losses.MSE,
        metrics=['AUC', 'acc'],
        run_eagerly=True
    )

    return model
