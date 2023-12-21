from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from utils.models.aspp_model import SaliencyNormalizationLayer, kld


def compare_models(
    model_paths: list[str],
    preprocess_funcs: list[Callable[[cv2.Mat], cv2.Mat]],
    images: list[tuple[str, str]]
):
    """Compares the models on the given images."""

    fig, axs = plt.subplots(len(images), len(model_paths) + 2)
    fig.set_size_inches(5 * len(model_paths) + 2, 3 * len(images))

    custom_objects = {
        'kl_divergence': tf.keras.losses.kld,
        'kld': kld,
        'SaliencyNormalizationLayer': SaliencyNormalizationLayer
    }

    models: list[tf.keras.Model] = [
        tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        for model_path in model_paths
    ]

    for i, (image_path, map_path) in enumerate(images):
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        ground_truth = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)

        axs[i, 0].axis('off')
        axs[i, 0].imshow(cv2.resize(image, (224, 224)))

        axs[i, 1].axis('off')
        axs[i, 1].imshow(cv2.resize(ground_truth, (224, 224)), 'gray')

        for model_index, model in enumerate(models):
            model_input = preprocess_funcs[model_index](image)

            model_output = model.predict(np.array([model_input]))

            output_image = model_output[0, :, :, :]

            axs[i, model_index + 2].axis('off')
            axs[i, model_index + 2].set_title('test')
            axs[i, model_index + 2].imshow(cv2.resize(output_image, (224, 224)), 'gray')

    plt.show()
