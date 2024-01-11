import tensorflow as tf


def correlation_coefficient(y_true, y_pred):
    y_true_flat = tf.keras.backend.flatten(y_true)
    y_pred_flat = tf.keras.backend.flatten(y_pred)

    mean_true = tf.keras.backend.mean(y_true_flat)
    mean_pred = tf.keras.backend.mean(y_pred_flat)

    covar = tf.keras.backend.mean((y_true_flat - mean_true) * (y_pred_flat - mean_pred))
    var_true = tf.keras.backend.var(y_true_flat)
    var_pred = tf.keras.backend.var(y_pred_flat)

    cc = covar / (tf.keras.backend.sqrt(var_true) * tf.keras.backend.sqrt(var_pred) + tf.keras.backend.epsilon())

    return cc


def similarity_metric(y_true, y_pred):
    y_true_density = y_true / tf.reduce_sum(y_true)
    y_pred_density = y_pred / tf.reduce_sum(y_pred)

    intersection = tf.reduce_sum(tf.minimum(y_true_density, y_pred_density))

    similarity = 2.0 * intersection / (
            tf.reduce_sum(y_true_density) + tf.reduce_sum(y_pred_density) + tf.keras.backend.epsilon())

    return similarity


def kld(y_true, y_pred, eps=1e-7):
    """This function computes the Kullback-Leibler divergence between ground
       truth saliency maps and their predictions. Values are first divided by
       their sum for each image to yield a distribution that adds to 1.

    Args:
        y_true (tensor, float32): A 4d tensor that holds the ground truth
                                  saliency maps with values between 0 and 255.
        y_pred (tensor, float32): A 4d tensor that holds the predicted saliency
                                  maps with values between 0 and 1.
        eps (scalar, float, optional): A small factor to avoid numerical
                                       instabilities. Defaults to 1e-7.

    Returns:
        tensor, float32: A 0D tensor that holds the averaged error.
    """

    sum_per_image = tf.reduce_sum(y_true, axis=(1, 2, 3), keepdims=True)
    y_true /= eps + sum_per_image

    sum_per_image = tf.reduce_sum(y_pred, axis=(1, 2, 3), keepdims=True)
    y_pred /= eps + sum_per_image
    loss = y_true * tf.math.log(eps + y_true / (eps + y_pred))
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=(1, 2, 3)))

    return loss
