import tensorflow as tf

from utils.preprocess import preprocess
from utils.visualize import compare_models

# folders_to_use = ["CSSD", "ECSSD", "MSRA-B", "MSRA10K_Imgs_GT", "THuR15000"]
#
# x_path = "../SP1_datasalicon/stimuli/imgs"
# y_path = "../SP1_datasalicon/saliency/imgs"
#
# train_model("aspp_salicon2.h5", x_path, y_path)

# test_model("saved_models/checkpoints/model.10.h5",
#            "../SP1_data/PseudoSaliency_avg_release/Images/MSRA10K_Imgs_GT/Imgs/0_7_7677.jpg",
#            "../SP1_data/PseudoSaliency_avg_release/Images/MSRA10K_Imgs_GT/Imgs/0_7_7677.jpg")


compare_models(
    [
        'saved_models/u-net.h5',
        'saved_models/deepgaze.h5',
        'saved_models/aspp_salicon2.h5',
        'saved_models/autoencoder2.h5',
        'saved_models/autoencoder_mse.h5'
    ],
    [
        preprocess((224, 224), tf.keras.applications.vgg19.preprocess_input),
        preprocess((256, 256), normalize=True),
        preprocess((224, 224), tf.keras.applications.vgg16.preprocess_input),
        preprocess((224, 224), tf.keras.applications.vgg19.preprocess_input),
        preprocess((224, 224), tf.keras.applications.vgg19.preprocess_input)
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
