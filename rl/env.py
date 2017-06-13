class EnvWrapper:
    def __init__(self, gym_env):
        self.env = gym_env
        # Restrict to euclidian spaces
        self.action_space_dim = self.env.action_space.shape[0]
        self.state_space_dim = self.env.observation_space.shape[0]
