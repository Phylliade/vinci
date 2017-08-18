import numpy as np
import gym.spaces


def spaces_grid(*spaces, definition=50):
    """
    Return a meshgrid covering the cartesian product of the given spaces

    :param spaces: Minimum one
    """
    low = np.concatenate([space.low for space in spaces], axis=0)
    high = np.concatenate([space.high for space in spaces], axis=0)
    dim = low.shape[0]
    axes = []

    for x in range(dim):
        axes.append(np.linspace(low[x], high[x], definition))

    return(np.meshgrid(*axes))


def merge_spaces(*spaces):
    """Merge the given spaces"""
    for space in spaces:
        if not isinstance(space, gym.spaces.Box):
            raise("Your given space is not of type Box")
    low = np.concatenate([space.low for space in spaces], axis=0)
    high = np.concatenate([space.high for space in spaces], axis=0)
    return gym.spaces.Box(low, high)
