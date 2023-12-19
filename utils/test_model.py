import os.path
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from config import *


def preprocess(image: cv2.Mat):
    resized = np.resize(image, (IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    return tf.keras.applications.vgg19.preprocess_input(resized)


def test_model(model_path, image_path, comparison_path):

    if os.path.exists(model_path):
        model:tf.keras.Model = tf.keras.models.load_model(model_path, custom_objects={"kl_divergence": tf.keras.losses.kld})

        test_im = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        input_im = preprocess(test_im)
        output = model.predict(np.array([input_im]))
        output = output[0, :, :, :]
        # output = cv2.GaussianBlur(output, (7, 7), 6)

        real_im = cv2.imread(comparison_path, cv2.IMREAD_GRAYSCALE)
        real_im = cv2.resize(real_im, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)

        fig, ax = plt.subplots(1, 3)
        ax[0].axis("off")
        ax[0].imshow(test_im)
        ax[1].imshow(output, 'gray')
        ax[1].axis("off")
        ax[2].imshow(real_im, cmap='gray')
        ax[2].axis("off")
        plt.show()
