from .hook import Hook
import pandas as pd


class ArrayHook(Hook):
    def _register_experiments(self, *args, **kwargs):
        super(ArrayHook, self)._register_experiments(*args, **kwargs)
        self.experiments_data = []
        self.endpoint = self.experiments.endpoint("data")

    def _register(self, *args, **kwargs):
        super(ArrayHook, self)._register(*args, **kwargs)
        self.run_rewards = []
        self.rewards = []

    def __call__(self):
        if self.agent.done:
            self.run_rewards.append(self.agent.episode_reward)

    def _run_call(self):
        if self.agent.run_done:
            self.rewards.append(self.run_rewards)
            self.run_rewards = []

    def _experiment_call(self):
        if self.experiment.done:
            self.experiments_data.append(self.rewards)
            self.rewards = []

    def _experiments_call(self):
        self.save()

    def save(self):
        rewards = pd.DataFrame(self.experiments_data)
        rewards.to_pickle(self.endpoint + "reward.p")


class TestArrayHook(ArrayHook):
    def __call__(self):
        # Only activate during testing
        if not self.agent.training:
            super(TestArrayHook, self).__call__()
