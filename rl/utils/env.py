import keras


def populate_env(env):
    # Restrict to euclidian spaces
    env.action_space.dim = env.action_space.shape[0]
    env.observation_space.dim = env.observation_space.shape[0]

    # Create the placeholders for the state and action
    # These are used as inputs for the Deep Networks, in keras
    env.state = keras.layers.Input(shape=(env.observation_space.dim,), name="state")
    env.action = keras.layers.Input(shape=(env.action_space.dim,), name="action")
    return(env)
