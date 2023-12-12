import keras.layers as layers
import numpy as np
import tensorflow as tf
from keras.layers import Resizing as Resize


def Gaussian_kernel(l=5, sig=1):
    """\
    creates gaussian kernel with side length l and a sigma of sig
    https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel / np.sum(kernel)


vgg16 = tf.keras.applications.vgg16.VGG16(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=(256, 256, 3), pooling=True)

for layer in vgg16.layers:

    names = ['block4_conv1',
             'block4_conv2',
             'block4_conv3',
             'block4_pool',
             'block5_conv1',
             'block5_conv2',
             'block5_conv3',
             'block5_pool']
    if layer.name not in names:
        layer.trainable = False
    if layer.name in names:
        layer.trainable = True

# """ Extend the base vgg16 model to a FCN-8 to generate fine-semantic map. """
# inputs = tf.keras.Input(shape=(256,256,3))
# x1 = tf.keras.applications.vgg16.preprocess_input(inputs)
x5 = vgg16.output  # 7*7*512

# The selections of the feature maps are slightly different from the original
# pepers because Tensroflow based Keras pre-built vgg16 doesn't allow access to
# the conv layers before relu activation.
# Instead, I chose conv4_3 after relu, and conv4_pool.
# The 'conv' prefix here is equiavalent to 'relu' in the original paper
# because all these layers go through relu activations.

x_conv4_3 = vgg16.get_layer('block4_conv3').output  # 28*28*512

x_conv4_pool = vgg16.get_layer('block4_pool').output  # 14*14*512

x_conv5_1 = vgg16.get_layer('block5_conv1').output  # 14*14*512

x_conv5_2 = vgg16.get_layer('block5_conv2').output  # 14*14*512

x_conv5_3 = vgg16.get_layer('block5_conv3').output  # 14*14*512

# resize all 5 layers into 128x128

feature_pyramid = [x_conv4_3, x_conv4_pool, x_conv5_1, x_conv5_2, x_conv5_3]

# feature_pyramid = [ x_conv5_1, x_conv5_2, x_conv5_3]


x6 = [Resize(128, 128)(x) for x in feature_pyramid]  # 128 x 128 x 512*3

x7 = layers.Concatenate(name='block7_cat1')(x6)

# Readout Network
l1 = tf.keras.regularizers.L1(0.001)
l2 = tf.keras.regularizers.L2(1e-2)
x8_1 = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', kernel_regularizer=l2, name='block8_conv1')(
    x7)  # 128*128*16
x8_2 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', kernel_regularizer=l2, name='block8_conv2')(
    x8_1)  # 128*128*32
x8_3 = tf.keras.layers.Conv2D(2, (1, 1), activation='relu', kernel_regularizer=l2, name='block8_conv3')(
    x8_2)  # 128*128*2
x8_4 = tf.keras.layers.Conv2D(1, (1, 1), activation='relu', kernel_regularizer=l2, name='block8_conv4')(
    x8_3)  # 128*128*1
x8_5 = Resize(256, 256, name='block8_rs')(x8_4)

# Gaussian Smoothing
gaussian_kernel = tf.convert_to_tensor(Gaussian_kernel(l=10, sig=5))
gaussian_kernel = tf.expand_dims(gaussian_kernel, axis=-1)
gaussian_kernel = tf.expand_dims(gaussian_kernel, axis=-1)
# x9 = tf.nn.conv2d(x8_5, gaussian_kernel, strides=[1, 1, 1, 1], padding='SAME', name='block9_conv1')
x9 = tf.keras.layers.GaussianNoise(8)
x9_1 = Resize(28, 28, name='block9_rs1')(x9)
# Add center_bias in form of log probability of the whole training set.
# center_bias = tf.keras.Input(shape=[256, 256, 1], name='block10_in')
# cb = Resize(28,28, name='block10_rs1')(center_bias)
# cb_2 = layers.Flatten(name='block10_flat1')(cb)
# p_cb = layers.Softmax(name='block10_soft1')(cb_2)
# def logFunc(x):
#     return tf.math.log(x)
# p_cb_2 = layers.Lambda(logFunc, name='block10_lambda')(p_cb)
# p_cb_3 = layers.Reshape((28,28,1), name='block10_rs')(p_cb_2)
# p_cb_3 = 0.0*cb
# x10 = layers.Add(name='block10_add')([x9_1, p_cb_3])
x = layers.BatchNormalization(name='block10_bn')(x9_1)
x = layers.Activation('sigmoid', name='block10_out')(x)

# Covnert to probability
# x = layers.Flatten()(x10)
# x = layers.Softmax()(x)
# x = layers.Reshape((28,28,1))(x)
# Resize
outputs = Resize(480, 640)(x)

model = tf.keras.Model(vgg16.input, outputs)

