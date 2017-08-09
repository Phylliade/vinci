import tensorflow as tf
import numpy as np


def gradient_inverter(gradient, p_min, p_max):
    """Gradient inverting as described in https://arxiv.org/abs/1511.04143"""
    delta = p_max - p_min
    if delta <= 0:
        raise(ValueError("p_max <= p_min"))

    inverted_gradient = tf.where(gradient >= 0, (p_max - gradient) / delta, (gradient - p_min) / delta)

    return(inverted_gradient)


def huber_loss(y_true, y_pred, clip_value):
    # Huber loss, see https://en.wikipedia.org/wiki/Huber_loss and
    # https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
    # for details.
    assert clip_value > 0.

    x = y_true - y_pred
    if np.isinf(clip_value):
        # Spacial case for infinity since Tensorflow does have problems
        # if we compare `K.abs(x) < np.inf`.
        return .5 * tf.square(x)

    condition = tf.abs(x) < clip_value
    squared_loss = .5 * tf.square(x)
    linear_loss = clip_value * (tf.abs(x) - .5 * clip_value)

    return tf.where(condition, squared_loss, linear_loss)  # condition, true, false
