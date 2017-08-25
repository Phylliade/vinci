from keras.models import model_from_config
import keras.optimizers as optimizers
# FIXME: Use only one of the two
import tensorflow as tf


def clone_model(model, custom_objects={}):
    # Requires Keras 1.0.7 since get_config has breaking changes.
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    clone.set_weights(model.get_weights())
    return clone


def clone_optimizer(optimizer):
    if type(optimizer) is str:
        return optimizers.get(optimizer)
    # Requires Keras 1.0.7 since get_config has breaking changes.
    params = dict([(k, v) for k, v in optimizer.get_config().items()])
    config = {
        'class_name': optimizer.__class__.__name__,
        'config': params,
    }
    if hasattr(optimizers, 'optimizer_from_config'):
        # COMPATIBILITY: Keras < 2.0
        clone = optimizers.optimizer_from_config(config)
    else:
        clone = optimizers.deserialize(config)
    return clone


def get_soft_target_model_ops(target_weights, source_weights, tau):
    ops = []
    for (target_weight, source_weight) in zip(target_weights, source_weights):
        ops.append(tf.assign(target_weight, tau * target_weight + (1. - tau) * source_weight))

    return(ops)


def get_soft_target_model_updates(target, source, tau):
    target_weights = target.trainable_weights + sum([l.non_trainable_weights for l in target.layers], [])
    source_weights = source.trainable_weights + sum([l.non_trainable_weights for l in source.layers], [])
    assert len(target_weights) == len(source_weights)

    # Create updates.
    updates = []
    for tw, sw in zip(target_weights, source_weights):
        updates.append((tw, tau * sw + (1. - tau) * tw))
    return updates


def get_object_config(o):
    if o is None:
        return None

    config = {
        'class_name': o.__class__.__name__,
        'config': o.get_config()
    }
    return config


class AdditionalUpdatesOptimizer(optimizers.Optimizer):
    def __init__(self, optimizer, additional_updates):
        super(AdditionalUpdatesOptimizer, self).__init__()
        self.optimizer = optimizer
        self.additional_updates = additional_updates

    def get_updates(self, params, constraints, loss):
        updates = self.optimizer.get_updates(params, constraints, loss)
        updates += self.additional_updates
        self.updates = updates
        return self.updates

    def get_config(self):
        return self.optimizer.get_config()
