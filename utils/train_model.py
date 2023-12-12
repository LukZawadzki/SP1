import keras.callbacks
import tensorflow as tf
from utils.aspp_model import create_aspp


def train_model(model_name, x_path, y_path):
    model = create_aspp()

    x_train, x_valid = (tf.keras.preprocessing.image_dataset_from_directory(
        x_path, labels=None, label_mode=None, image_size=[224, 224],
        batch_size=6, validation_split=0.2, subset="both", shuffle=False))

    x_train = x_train.map(tf.keras.applications.vgg16.preprocess_input)
    x_valid = x_valid.map(tf.keras.applications.vgg16.preprocess_input)

    y_train, y_valid = tf.keras.preprocessing.image_dataset_from_directory(
        y_path, labels=None, label_mode=None, image_size=[224, 224], batch_size=6,
        color_mode="grayscale", validation_split=0.2, subset="both", shuffle=False)

    train_set = tf.data.Dataset.zip((x_train, y_train))
    valid_set = tf.data.Dataset.zip((x_valid, y_valid))

    saving_models = keras.callbacks.ModelCheckpoint("./saved_models/checkpoints/model.{epoch:02d}.h5")
    model.fit(train_set, validation_data=valid_set, epochs=10, callbacks=[saving_models])
    model.save("saved_models/"+model_name)
    return model
