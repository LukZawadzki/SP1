from typing import Literal, Callable

import tensorflow as tf

from .deepgaze import create_deepgaze
from .encoder_decoder import create_encoder_decoder
from .msi_net_model import create_msi_net
from .unet import create_unet

__all__ = (
    'create_deepgaze',
    'create_encoder_decoder',
    'create_msi_net',
    'create_unet',
    'ModelTypes',
    'MODELS'
)

ModelTypes = Literal['DeepGaze', 'MSI-Net', 'Encoder-Decoder', 'U-Net']

MODELS: dict[ModelTypes, Callable[[], tf.keras.Model]] = {
    'DeepGaze': create_deepgaze,
    'MSI-Net': create_msi_net,
    'Encoder-Decoder': create_encoder_decoder,
    'U-Net': create_unet
}
