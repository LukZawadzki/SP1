from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import INPUT_IMAGE_SIZE, CUSTOM_OBJECTS


def compare_models(
    model_paths: list[str],
    preprocess_funcs: list[Callable[[cv2.Mat], cv2.Mat]],
    images: list[tuple[str, str]]
):
    """Compares the models on the given images."""

    fig, axs = plt.subplots(len(images), len(model_paths) + 2)
    # fig.set_size_inches(5 * len(model_paths) + 2, 3 * len(images))

    models: list[tf.keras.Model] = [
        tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)
        for model_path in model_paths
    ]

    for i, (image_path, map_path) in enumerate(images):
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        ground_truth = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)

        axs[i, 0].axis('off')
        axs[i, 0].imshow(cv2.resize(image, INPUT_IMAGE_SIZE))

        axs[i, 1].axis('off')
        axs[i, 1].imshow(cv2.resize(ground_truth, INPUT_IMAGE_SIZE), 'gray')

        for model_index, model in enumerate(models):
            model_input = preprocess_funcs[model_index](image)

            model_output = model.predict(np.array([model_input]))

            output_image = model_output[0, :, :, :]

            axs[i, model_index + 2].axis('off')
            axs[i, model_index + 2].imshow(cv2.resize(output_image, INPUT_IMAGE_SIZE), 'gray')

    plt.show()
