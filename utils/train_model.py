import keras.callbacks
import tensorflow as tf
from utils.models.aspp_model import create_aspp, SaliencyNormalizationLayer, kld
from config import *


def train_model(model_name, x_path, y_path, path=None):
    x_train, x_valid = (tf.keras.preprocessing.image_dataset_from_directory(
        x_path, labels=None, label_mode=None, image_size=IMAGE_SIZE,
        batch_size=1, validation_split=0.33, subset="both", seed=35, interpolation="area"))
    # x_train = tf.keras.preprocessing.image_dataset_from_directory(
    #     x_path+"train", labels=None, label_mode=None, image_size=IMAGE_SIZE,
    #     batch_size=1, interpolation="bicubic", seed=22)
    # x_valid = tf.keras.preprocessing.image_dataset_from_directory(
    #     x_path+"val", labels=None, label_mode=None, image_size=IMAGE_SIZE,
    #     batch_size=1, interpolation="bicubic", seed=22)

    # x_train = x_train.map(tf.keras.applications.vgg16.preprocess_input).map(lambda x: x/255)
    # x_valid = x_valid.map(tf.keras.applications.vgg16.preprocess_input).map(lambda x: x/255)
    x_train = x_train.map(tf.keras.applications.vgg16.preprocess_input)
    x_valid = x_valid.map(tf.keras.applications.vgg16.preprocess_input)

    y_train, y_valid = tf.keras.preprocessing.image_dataset_from_directory(
        y_path, labels=None, label_mode=None, image_size=IMAGE_SIZE, batch_size=1,
        color_mode="grayscale", validation_split=0.33, subset="both", seed=35, interpolation="area")

    # y_train = tf.keras.preprocessing.image_dataset_from_directory(
    #     y_path + "train", labels=None, label_mode=None, image_size=IMAGE_SIZE, batch_size=1,
    #     color_mode="grayscale", interpolation="bicubic", seed=22)
    # y_valid = tf.keras.preprocessing.image_dataset_from_directory(
    #     y_path + "val", labels=None, label_mode=None, image_size=IMAGE_SIZE, batch_size=1,
    #     color_mode="grayscale", interpolation="bicubic", seed=22)

    # y_train = y_train.map(lambda x: x/255)
    # y_valid = y_valid.map(lambda x: x/255)

    train_set = tf.data.Dataset.zip((x_train, y_train))
    valid_set = tf.data.Dataset.zip((x_valid, y_valid))

    saving_models = keras.callbacks.ModelCheckpoint("./saved_models/checkpoints/model.{epoch:02d}.h5")
    if path:
        model = tf.keras.models.load_model(path, custom_objects={"kl_divergence": tf.keras.losses.kld,
                                                                 "kld": kld,
                                           "SaliencyNormalizationLayer": SaliencyNormalizationLayer})
    else:
        model = create_aspp()
    model.fit(train_set, validation_data=valid_set, epochs=10, callbacks=[saving_models])
    model.save("saved_models/"+model_name)
    return model
