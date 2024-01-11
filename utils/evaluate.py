from config import INPUT_IMAGE_SIZE, CUSTOM_OBJECTS
from utils.metrics import *


def evaluate(model_path: str, path: str, preprocess=tf.keras.applications.vgg19, normalize_x=False, normalize_y=False):
    """Evaluates the model."""

    original_model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)

    new_model = tf.keras.models.clone_model(original_model)
    new_model.set_weights(original_model.get_weights())
    new_model.compile(
        optimizer=original_model.optimizer,
        loss=original_model.loss,
        metrics=[tf.keras.metrics.AUC(), kld, correlation_coefficient, similarity_metric]
    )

    x = tf.keras.preprocessing.image_dataset_from_directory(
        path+"stimuli/",
        labels=None,
        label_mode=None,
        image_size=INPUT_IMAGE_SIZE,
        interpolation="area",
        batch_size=1
    )

    if normalize_x:
        x = x.map(preprocess.preprocess_input).map(lambda z: z / 255)
    else:
        x = x.map(preprocess.preprocess_input)

    y = tf.keras.preprocessing.image_dataset_from_directory(
        path+"saliency/",
        labels=None,
        label_mode=None,
        image_size=INPUT_IMAGE_SIZE,
        batch_size=1,
        color_mode="grayscale",
        interpolation="area"
    )

    if normalize_y:
        y = y.map(lambda z: z / 255)

    print(new_model.evaluate(tf.data.Dataset.zip((x, y))))
