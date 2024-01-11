import keras.callbacks
import tensorflow as tf

from config import INPUT_IMAGE_SIZE, CUSTOM_OBJECTS
from models import ModelTypes, MODELS


def train_model(
    save_as: str,
    x_path: str,
    y_path: str,
    saved_model_path: str = None,
    epochs: int = 10,
    batch_size: int = 8,
    validation_split: float = 0.25,
    model_type: ModelTypes = 'MSI-NET',
    preprocess=tf.keras.applications.imagenet_utils,
    normalize_x: bool = False,
    normalize_y: bool = False,
    seed: int = 42
) -> tf.keras.Model:
    """
    Use it to train your model from scratch or to train some pretrained one
    :param save_as: name the model will be given when saving after training
    :param x_path: relative path to input x data
    :param y_path: relative path to real y data
    :param saved_model_path: relative path to some saved model, if you want to train some pretrained earlier model, must be None if you want to create new model
    :param epochs: how many epochs in training
    :param batch_size: the batch size for training
    :param validation_split: the validation split
    :param model_type: one of "DeepGaze", "MSI-Net", "Encoder-Decoder", "U-Net", to create new model of given type
    :param preprocess: backbone used in model to preprocess images (tf.keras.applications)
    :param normalize_x: normalize input to range (0, 1) from (0, 255)
    :param normalize_y: normalize input to range (0, 1) from (0, 255)
    :param seed: the seed to use for random dataset generating
    :return:
    """

    x_train, x_valid = tf.keras.preprocessing.image_dataset_from_directory(
        x_path,
        labels=None,
        label_mode=None,
        image_size=INPUT_IMAGE_SIZE,
        batch_size=batch_size,
        validation_split=validation_split,
        subset="both",
        seed=seed,
        interpolation="area"
    )

    if normalize_x:
        x_train = x_train.map(preprocess.preprocess_input).map(lambda x: x / 255)
        x_valid = x_valid.map(preprocess.preprocess_input).map(lambda x: x / 255)
    else:
        x_train = x_train.map(preprocess.preprocess_input)
        x_valid = x_valid.map(preprocess.preprocess_input)

    y_train, y_valid = tf.keras.preprocessing.image_dataset_from_directory(
        y_path,
        labels=None,
        label_mode=None,
        image_size=INPUT_IMAGE_SIZE,
        batch_size=batch_size,
        color_mode="grayscale",
        validation_split=validation_split,
        subset="both",
        seed=seed,
        interpolation="area"
    )

    if normalize_y:
        y_train = y_train.map(lambda x: x / 255)
        y_valid = y_valid.map(lambda x: x / 255)

    train_set = tf.data.Dataset.zip((x_train, y_train))
    valid_set = tf.data.Dataset.zip((x_valid, y_valid))

    saving_models = keras.callbacks.ModelCheckpoint("./saved_models/checkpoints/model.{epoch:02d}.h5")
    if saved_model_path:
        model = tf.keras.models.load_model(saved_model_path, custom_objects=CUSTOM_OBJECTS)
    else:
        model = MODELS.get(model_type)()

    model.fit(train_set, validation_data=valid_set, epochs=epochs, callbacks=[saving_models])
    model.save(save_as)
    return model
