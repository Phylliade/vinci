from rl.runtime.agent import Agent
from rl.runtime.experiment import DefaultExperiment
from rl.utils.printer import print_epoch
import numpy as np
import pickle


class OmniscientAgent(Agent):
    def __init__(self, experiment, **kwargs):
        if experiment is None:
            self.experiment = DefaultExperiment(use_tf=False)
        else:
            self.experiment = experiment

        self.replay_buffer = []
        # Finish with agent initialization since Hook initialization can depend on custom GEPAgent variables
        super(OmniscientAgent, self).__init__(experiment=self.experiment, **kwargs)

    def test(self, steps, render=False, observation_space_low=None, observation_space_high=None, action_space_low=None, action_space_high=None):
        self.env.reset()
        if observation_space_low is None:
            observation_space_low = self.env.observation_space.low
        if observation_space_high is None:
            observation_space_high = self.env.observation_space.high
        if action_space_high is None:
            action_space_high = self.env.action_space.high
        if action_space_low is None:
            action_space_low = self.env.action_space.low

        for epoch in range(1, steps + 1):
            print_epoch(epoch, steps)
            # Pick a random state
            observation_pre = np.random.uniform(observation_space_low, observation_space_high, size=self.env.observation_space.shape)

            # Set it in the environment
            # FIXME: Expose set_state directly in the environment root
            self.env.env.set_state(observation_pre)

            # Simulate from this state with a random action
            random_action = np.random.uniform(action_space_low, action_space_high, size=self.env.action_space.shape)
            observation_post, reward, done, info = self.env.step(random_action)
            self.replay_buffer.append([observation_pre, random_action, reward, observation_post, done])

            if render:
                self.env.render()

    def dump_memory(self, file):
        with open(file, "wb") as fd:
            pickle.dump(self.replay_buffer, fd)
