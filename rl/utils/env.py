import keras
import numpy as np


def populate_env(env):
    # Restrict to euclidian spaces
    env.action_space.dim = env.action_space.shape[0]
    env.observation_space.dim = env.observation_space.shape[0]

    # Avoid having infinite bounds
    for array in [env.observation_space.low, env.observation_space.high]:
        if (abs(array) == np.inf).any():
            # Clip to min and max values, except infinite
            # Default to -1 and 1 if the array is only filled with inf
            low = np.min(env.observation_space.low)
            if low == -np.inf:
                low = -1
            high = np.max(env.observation_space.low)
            if high == np.inf:
                high = 1

            array[:] = np.clip(array, low, high)

    # Create the placeholders for the state and action
    # These are used as inputs for the Deep Networks, in keras
    env.state = keras.layers.Input(shape=(env.observation_space.dim,), name="state")
    env.action = keras.layers.Input(shape=(env.action_space.dim,), name="action")
    return(env)
