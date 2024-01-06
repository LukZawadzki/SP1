import tensorflow as tf
from utils.metrics import *
from utils.models.aspp_model import create_aspp, SaliencyNormalizationLayer, kld
from config import IMAGE_SIZE


def evaluate(model_path: str, path: str, preprocess=tf.keras.applications.vgg19, normalize_x=False, normalize_y=False):
    original_model = tf.keras.models.load_model(model_path,
                                                custom_objects={"kl_divergence": tf.keras.losses.kld, "kld": kld,
                                                                "SaliencyNormalizationLayer": SaliencyNormalizationLayer})
    new_model = tf.keras.models.clone_model(original_model)
    new_model.set_weights(original_model.get_weights())
    new_model.compile(optimizer=original_model.optimizer, loss=original_model.loss,
                      metrics=[tf.keras.metrics.AUC(), kld, correlation_coefficient, similarity_metric])

    # new_model.add_metric(nss_metric, name="nss")
    # new_model.add_metric(correlation_coefficient, name="cc")
    # new_model.add_metric(information_gain, name="ig")
    # new_model.add_metric(tf.keras.metrics.AUC, name="auc")
    # new_model.add_metric(kld, name="kld")

    x = (tf.keras.preprocessing.image_dataset_from_directory(
        path+"stimuli/", labels=None, label_mode=None, image_size=IMAGE_SIZE, interpolation="area", batch_size=1))

    if normalize_x:
        x = x.map(preprocess.preprocess_input).map(lambda z: z / 255)
    else:
        x = x.map(preprocess.preprocess_input)

    y = tf.keras.preprocessing.image_dataset_from_directory(
        path+"saliency/", labels=None, label_mode=None, image_size=IMAGE_SIZE, batch_size=1,
        color_mode="grayscale", interpolation="area")

    if normalize_y:
        y = y.map(lambda z: z / 255)

    print(new_model.evaluate(tf.data.Dataset.zip((x, y))))


evaluate("../saved_models/encoder_decoder_salicon_msrab_msra_dut.h5",
         "../../SP1_datamit1003/", preprocess=tf.keras.applications.vgg16, normalize_y=True)
