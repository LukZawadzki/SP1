import math

import keras.initializers.initializers_v2
import keras.layers as layers
from keras.applications.vgg16 import VGG16
from keras.regularizers import l2
from keras import constraints, regularizers, activations
from config import *
import tensorflow as tf
import keras.backend as K


class EltWiseProduct(layers.Layer):
    def __init__(self, downsampling_factor=10, init='glorot_uniform', activation='linear',
                 weights=None, W_regularizer=None, activity_regularizer=None,
                 W_constraint=None, input_dim=None, **kwargs):

        self.downsampling_factor = downsampling_factor
        self.init = keras.initializers.get(init)
        self.activation = activations.get(activation)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)

        self.initial_weights = weights

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)

        self.input_spec = [layers.InputSpec(ndim=4)]
        super(EltWiseProduct, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.init([s // self.downsampling_factor for s in input_shape[2:]])

        # self.trainable_weights = [self.W]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint

    def get_output_shape_for(self, input_shape):
        return input_shape

    def call(self, x, mask=None, **kwargs):

        output = x*bilinear_upsampling(K.expand_dims(K.expand_dims(1 + self.W, 0), 0), self.downsampling_factor, 1, 1)
        return output

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'output_dim': self.input_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'input_dim': self.input_dim,
                  'downsampling_factor': self.downsampling_factor}
        base_config = super(EltWiseProduct, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def create_ml_net():
    downsampling_factor_product = 10
    downsampling_factor_net = 8
    img_rows = IMAGE_SIZE[0]
    img_cols = IMAGE_SIZE[1]

    backbone = VGG16(False, 'imagenet', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), pooling=None)

    new_backbone = [layers.InputLayer((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))]
    pooling_layers = []

    for i in range(1, len(backbone.layers)):
        previous_layer = new_backbone[i - 1]
        if isinstance(previous_layer, layers.InputLayer):
            output = previous_layer.output
        else:
            output = previous_layer
        backbone_layer = backbone.get_layer(index=i)
        if "pool" in backbone_layer.name:
            if backbone_layer.name == "block5_pool":
                break
            if i >= 13:
                new_layer = layers.MaxPooling2D(pool_size=2, strides=1, padding="same", name=backbone_layer.name)(output)
            else:
                new_layer = layers.MaxPooling2D(pool_size=2, strides=2, padding="same", name=backbone_layer.name)(output)
            pooling_layers.append(new_layer)
        else:
            old_layer = backbone_layer
            new_layer = layers.Conv2D(old_layer.filters, old_layer.kernel_size, old_layer.strides,
                                      padding="same", dilation_rate=old_layer.dilation_rate, activation="relu",
                                      weights=old_layer.get_weights(), name=backbone_layer.name)(output)
        new_backbone.append(new_layer)

    concat = tf.concat([pooling_layers[2], pooling_layers[3], new_backbone[-1]], axis=-1)
    dropout = layers.Dropout(0.5)(concat)

    int_conv = layers.Conv2D(64, 3, 1, kernel_initializer='glorot_normal',
                             activation='relu', padding='same')(dropout)

    pre_final_conv = layers.Conv2D(1, 1, 1,
                                   kernel_initializer='glorot_normal', activation='relu')(int_conv)

    rows_elt = math.ceil(img_rows / downsampling_factor_net) // downsampling_factor_product
    cols_elt = math.ceil(img_cols / downsampling_factor_net) // downsampling_factor_product
    eltprod = EltWiseProduct(init='zero', W_regularizer=l2(1 / (rows_elt * cols_elt)))(pre_final_conv)

    output_ml_net = layers.Activation('relu')(pre_final_conv)

    sgd = tf.keras.optimizers.SGD(learning_rate=1e-3, decay=0.0005, momentum=0.9, nesterov=True)
    model = tf.keras.models.Model(inputs=new_backbone[0].input, outputs=output_ml_net)
    model.compile(optimizer=sgd, loss="mse", metrics=["AUC", "mse", "accuracy"])

    for layer in model.layers:
        if "block" in layer.name:
            layer.trainable = False
    print(model.summary())
    return model


def loss(y_true, y_pred):
    shape_r_gt = int(math.ceil(IMAGE_SIZE[0] / 8))
    shape_c_gt = int(math.ceil(IMAGE_SIZE[1] / 8))
    max_y = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)), shape_r_gt, axis=-1)), shape_c_gt, axis=-1)
    return K.mean(K.square((y_pred / max_y) - y_true) / (1 - y_true + 0.1))
