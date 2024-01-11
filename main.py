from config import INPUT_IMAGE_SIZE
from utils.download import download_dutomron
from utils.preprocess import preprocess
from utils.test_model import test_model
from utils.train_model import train_model
from utils.visualize import compare_models


def example_dataset_download():
    """Example usage of downloading a dataset."""

    download_dutomron('datasets/')


def example_train_model():
    """Example usage of training a model."""

    train_model(
        'saved_models/encoder-decoder.h5',
        'datasets/dutomron/stimuli',
        'datasets/dutomron/saliency',
        model_type='Encoder-Decoder'
    )


def example_test_model():
    """Example usage of testing a model."""

    test_model(
        'saved_models/encoder-decoder.h5',
        'example.png',
    )


def example_visual_model_comparison():
    """Example usage of visually comparing model outputs."""

    compare_models(
        [
            'saved_models/u-net.h5',
            'saved_models/msi-net.h5',
            'saved_models/encoder-decoder.h5',
            'saved_models/deepgaze.h5'
        ],
        [
            preprocess(INPUT_IMAGE_SIZE),
            preprocess(INPUT_IMAGE_SIZE),
            preprocess(INPUT_IMAGE_SIZE),
            preprocess((256, 256), normalize=True)
        ],
        [
            (
                'datasets/salicon/stimuli/val/COCO_val2014_000000002640.jpg',
                'datasets/salicon/saliency/val/COCO_val2014_000000002640.png'
            ),
            (
                'datasets/salicon/stimuli/val/COCO_val2014_000000008711.jpg',
                'datasets/salicon/saliency/val/COCO_val2014_000000008711.png'
            ),
            (
                'datasets/salicon/stimuli/train/COCO_train2014_000000002007.jpg',
                'datasets/salicon/saliency/train/COCO_train2014_000000002007.png'
            ),
            (
                'datasets/salicon/stimuli/val/COCO_val2014_000000017482.jpg',
                'datasets/salicon/saliency/val/COCO_val2014_000000017482.png'
            ),
            (
                'datasets/Images/ECSSD/images/0552.jpg',
                'datasets/Maps/ECSSD/images/0552.jpg'
            ),
        ]
    )
