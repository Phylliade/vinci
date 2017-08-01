import tensorflow as tf


def gradient_inverter(gradient, p_min, p_max):
    """Gradient inverting as described in https://arxiv.org/abs/1511.04143"""
    delta = p_max - p_min
    if delta <= 0:
        raise(ValueError("p_max <= p_min"))

    inverted_gradient = tf.where(gradient >= 0, (p_max - gradient) / delta, (gradient - p_min) / delta)

    return(inverted_gradient)
