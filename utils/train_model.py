import keras.callbacks
import tensorflow as tf
from utils.models.aspp_model import create_aspp, SaliencyNormalizationLayer, kld
from utils.models.unet import create_unet
from utils.models.deepgaze import create_deepgaze
from utils.models.mlnet import create_ml_net
from utils.models.autoencoder import create_autoencoder
from config import *


def train_model(model_name, x_path, y_path, saved_model_path=None, epochs=10, model_type="aspp",
                preprocess=tf.keras.applications.vgg16, normalize_x=False, normalize_y=False):
    """
    Use it to train your model from scratch or to train some pretrained one
    :param model_name: name the model will be given when saving after training
    :param x_path: relative path to input x data
    :param y_path: relative path to real y data
    :param saved_model_path: relative path to some saved model, if you want to train some pretrained earlier model, must be None if you want to create new model
    :param epochs: how many epochs in training
    :param model_type: one of "aspp", "mlnet", "deepgaze", "decoder", "unet" to create new model of given type
    :param preprocess: backbone used in model to preprocess images (tf.keras.applications)
    :param normalize_x: normalize input to range (0, 1) from (0, 255)
    :param normalize_y: normalize input to range (0, 1) from (0, 255)
    :return:
    """
    x_train, x_valid = (tf.keras.preprocessing.image_dataset_from_directory(
        x_path, labels=None, label_mode=None, image_size=IMAGE_SIZE,
        batch_size=1, validation_split=0.33, subset="both", seed=35, interpolation="area"))

    if normalize_x:
        x_train = x_train.map(preprocess.preprocess_input).map(lambda x: x / 255)
        x_valid = x_valid.map(preprocess.preprocess_input).map(lambda x: x / 255)
    else:
        x_train = x_train.map(preprocess.preprocess_input)
        x_valid = x_valid.map(preprocess.preprocess_input)

    y_train, y_valid = tf.keras.preprocessing.image_dataset_from_directory(
        y_path, labels=None, label_mode=None, image_size=IMAGE_SIZE, batch_size=1,
        color_mode="grayscale", validation_split=0.33, subset="both", seed=35, interpolation="area")

    if normalize_y:
        y_train = y_train.map(lambda x: x / 255)
        y_valid = y_valid.map(lambda x: x / 255)

    train_set = tf.data.Dataset.zip((x_train, y_train))
    valid_set = tf.data.Dataset.zip((x_valid, y_valid))

    saving_models = keras.callbacks.ModelCheckpoint("./saved_models/checkpoints/model.{epoch:02d}.h5")
    if saved_model_path:
        model = tf.keras.models.load_model(saved_model_path,
                                           custom_objects={"kl_divergence": tf.keras.losses.kld, "kld": kld,
                                                           "SaliencyNormalizationLayer": SaliencyNormalizationLayer})
    else:
        if model_type.lower() == "aspp":
            model = create_aspp()
        elif model_type.lower() == "deepgaze":
            model = create_deepgaze()
        elif model_type.lower() == "mlnet":
            model = create_ml_net()
        elif model_type.lower() == "unet":
            model = create_unet()
        elif model_type.lower() == "decoder":
            model = create_autoencoder()
        else:
            raise Exception("Model type not found")

    model.fit(train_set, validation_data=valid_set, epochs=epochs, callbacks=[saving_models])
    model.save("saved_models/" + model_name)
    return model
