from rl.runtime.agent import Agent
from rl.runtime.experiment import DefaultExperiment
from rl.utils.printer import print_epoch
import numpy as np
import pickle


class OmniscientAgent(Agent):
    def __init__(self, environment, experiment, **kwargs):
        self.environment = environment

        if experiment is None:
            self.experiment = DefaultExperiment(use_tf=False)
        else:
            self.experiment = experiment

        self.replay_buffer = []
        # Finish with agent initialization since Hook initialization can depend on custom GEPAgent variables
        super(OmniscientAgent, self).__init__(experiment=self.experiment, **kwargs)

    def test(self, steps, render=False):
        self.environment.reset()

        for epoch in range(1, steps + 1):
            print_epoch(epoch, steps)
            # Pick a random state
            observation_pre = np.random.uniform(self.environment.observation_space.low, self.environment.observation_space.high, size=self.environment.observation_space.shape)

            # Set it in the environment
            # FIXME: Expose set_state directly in the environment root
            self.environment.env.set_state(observation_pre)

            # Simulate from this state with a random action
            random_action = np.random.uniform(self.environment.action_space.low, self.environment.action_space.high, size=self.environment.action_space.shape)
            observation_post, reward, done, info = self.environment.step(random_action)
            self.replay_buffer.append([observation_pre, random_action, reward, observation_post, done])

            if render:
                self.environment.render()

    def dump_memory(self, file):
        with open(file, "wb") as fd:
            pickle.dump(self.replay_buffer, fd)
