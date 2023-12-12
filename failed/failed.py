import os
import tensorflow as tf
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.vgg19 import VGG19
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

folders_to_use = ["CSSD", "ECSSD", "MSRA-B", "MSRA10K_Imgs_GT", "THuR15000"]

# resnet = ResNet50V2(False, 'imagenet', None, input_shape=(224, 224, 3), pooling=None)
resnet = VGG19(False, 'imagenet', None, input_shape=(224, 224, 3), pooling=None)

for layer in resnet.layers:
    layer.trainable = False

l2 = tf.keras.regularizers.L2(1e-2)

layer_norm_1 = tf.keras.layers.LayerNormalization(name="layer_norm_1")(resnet.output)
conv_1 = tf.keras.layers.Conv2D(16,(1,1), activation='relu', kernel_regularizer=l2, name='block8_conv1')(layer_norm_1)
dense_1 = tf.keras.layers.Dense(8, name="dense_1", activation=tf.keras.activations.softplus)(conv_1)

layer_norm_2 = tf.keras.layers.LayerNormalization(name="layer_norm_2")(dense_1)
conv_2 = tf.keras.layers.Conv2D(32,(1,1), activation='relu', kernel_regularizer=l2, name='block8_conv2')(layer_norm_2)
dense_2 = tf.keras.layers.Dense(16, name="dense_2", activation=tf.keras.activations.softplus)(conv_2)

layer_norm_3 = tf.keras.layers.LayerNormalization(name="layer_norm_3")(dense_2)
conv_3 = tf.keras.layers.Conv2D(64,(1,1), activation ='relu', kernel_regularizer=l2, name='block8_conv3')(layer_norm_3)
dense_3 = tf.keras.layers.Dense(1, name="dense_3", activation=tf.keras.activations.softplus,)(conv_3)

layer_norm_4 = tf.keras.layers.LayerNormalization(name="layer_norm_4")(dense_3)
conv_4 = tf.keras.layers.Conv2D(64,(1,1), activation ='relu', kernel_regularizer=l2, name='block8_conv4')(layer_norm_4)
dense_4 = tf.keras.layers.Dense(128, name="dense_4", activation=tf.keras.activations.softplus,)(conv_4)

layer_norm_5 = tf.keras.layers.LayerNormalization(name="layer_norm_5")(dense_4)
conv_5 = tf.keras.layers.Conv2D(128,(1,1), activation ='relu', kernel_regularizer=l2, name='block8_conv5')(layer_norm_5)
dense_5 = tf.keras.layers.Dense(16, name="dense_5", activation=tf.keras.activations.softplus,)(conv_5)

layer_norm_6 = tf.keras.layers.LayerNormalization(name="layer_norm_6")(dense_5)
conv_6 = tf.keras.layers.Conv2D(64,(1,1), activation ='relu', kernel_regularizer=l2, name='block8_conv6')(layer_norm_6)
dense_6 = tf.keras.layers.Dense(1, name="dense_6", activation=tf.keras.activations.softplus,)(conv_6)

resize = tf.keras.layers.Resizing(480, 640)(dense_6)
# blur = tf.keras.layers.GaussianNoise(8)(resize)
softmax = tf.keras.layers.Softmax()(resize)

model = tf.keras.Model(inputs=resnet.input, outputs=softmax)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='mse',
              metrics=[tf.keras.metrics.KLD, "AUC", "accuracy"])


def preprocess(image: cv2.Mat):
    resized = np.resize(image, (224, 224, 3))
    return tf.keras.applications.resnet_v2.preprocess_input(resized)


processed_x = []
y = []

x_path = "../PseudoSaliency_avg_release/Images/CSSD/images/"
y_path = "../PseudoSaliency_avg_release/Maps/CSSD/images/"

for file in os.listdir(x_path):
    if file.endswith(".jpg"):
        image = cv2.imread(x_path+file)
        processed_x.append(preprocess(image))
for file in os.listdir(y_path):
    if file.endswith(".jpg"):
        y.append(cv2.imread(y_path+file, cv2.IMREAD_GRAYSCALE))

processed_x = np.array(processed_x)
y = np.array(y)
# X_train, X_test, y_train, y_test = train_test_split(processed_x, y, test_size=0.2)

# processed_x = tf.keras.preprocessing.image_dataset_from_directory("./PseudoSaliency_avg_release/Images/MSRA10K_Imgs_GT/Imgs/")

model.fit(processed_x, y, validation_split=0.2, epochs=25, steps_per_epoch=40, validation_steps=25)
model.save("model.h5")

# print(model.summary())
