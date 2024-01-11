import os.path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from utils.preprocess import preprocess

from config import INPUT_IMAGE_SIZE, CUSTOM_OBJECTS


def test_model(model_path: str, image_path: str, ground_truth_path: str | None = None):
    """
    :param model_path: path to trained model
    :param image_path: path to test image
    :param ground_truth_path: path to the ground truth image, if applicable
    """

    if not os.path.exists(model_path):
        print('Model path doesn\'t exist')
        return

    model: tf.keras.Model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)

    test_im = cv2.imread(image_path)
    input_im = preprocess(INPUT_IMAGE_SIZE)(test_im)
    output = model.predict(np.array([input_im]))
    output = output[0, :, :, :]

    ground_truth = None
    if ground_truth_path is not None:
        ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        ground_truth = cv2.resize(ground_truth, INPUT_IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)

    test_image_to_show = cv2.cvtColor(
        cv2.resize(test_im, INPUT_IMAGE_SIZE, interpolation=cv2.INTER_LINEAR),
        cv2.COLOR_BGR2RGB
    )

    fig, ax = plt.subplots(1, 3 if ground_truth_path is not None else 2)
    ax[0].axis("off")
    ax[0].imshow(test_image_to_show)
    if ground_truth_path is not None:
        ax[1].imshow(ground_truth, 'gray')
        ax[1].axis("off")
        ax[2].imshow(output, 'gray')
        ax[2].axis("off")
    else:
        ax[1].imshow(output, 'gray')
        ax[1].axis("off")

    plt.show()
