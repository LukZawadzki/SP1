import tensorflow as tf

from utils.layers import SaliencyNormalizationLayer
from utils.metrics import kld

__all__ = (
    'INPUT_IMAGE_SIZE',
    'CUSTOM_OBJECTS'
)

INPUT_IMAGE_SIZE = (224, 224)

CUSTOM_OBJECTS = {
    "kl_divergence": tf.keras.losses.kld,
    "kld": kld,
    "SaliencyNormalizationLayer": SaliencyNormalizationLayer
}
