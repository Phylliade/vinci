import keras


class EnvWrapper:
    def __init__(self, gym_env):
        self.env = gym_env
        # Restrict to euclidian spaces
        self.action_space_dim = self.env.action_space.shape[0]
        self.state_space_dim = self.env.observation_space.shape[0]

        # Create the placeholders for the state and action
        # These are used as inputs for the Deep Networks, in keras
        self.state = keras.layers.Input(shape=(self.state_space_dim,), name="state")
        self.action = keras.layers.Input(shape=(self.action_space_dim,), name="action")
