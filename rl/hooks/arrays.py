from .hook import Hook
import pandas as pd


class ArrayHook(Hook):
    def _register_experiments(self, *args, **kwargs):
        super(ArrayHook, self)._register_experiments(*args, **kwargs)
        self.experiments_data = []
        self.endpoint = self.experiments.endpoint("data")
        self.episode_index = []
        self.experiment_index = []
        self.training_run = []
        self.training_experiment = []

    def _register(self, *args, **kwargs):
        super(ArrayHook, self)._register(*args, **kwargs)
        self.run_rewards = []
        self.experiment_rewards = []

    def __call__(self):
        if self.agent.done:
            self.run_rewards.append(self.agent.episode_reward)

    def _run_call(self):
        if self.agent.run_done:
            self.experiment_rewards.append(self.run_rewards)
            self.run_rewards = []
            self.episode_index.append(self.agent.episode)
            self.training_run.append(self.agent.training)

    def _experiment_call(self):
        if self.experiment.done:
            self.experiments_data.append(self.experiment_rewards)
            self.experiment_rewards = []
            self.experiment_index.append(self.experiments.experiment_count)
            self.training_experiment.append(self.training_run)
            self.training_run = []
            # Save every 5 experiments
            if (self.experiments.experiment_count % 5 == 0):
                self.save()

    def _experiments_call(self):
        self.save()

    def save(self):
        print("Saving data")
        rewards = pd.DataFrame(self.experiments_data, index=self.experiment_index)
        rewards.columns.name = "runs"
        rewards.index.name = "experiment"
        rewards.to_pickle(self.endpoint + "reward.p")

        episodes = pd.DataFrame(self.episode_index, index=self.episode_index)
        episodes.to_pickle(self.endpoint + "episodes.p")

        training = pd.DataFrame(self.training_experiment, index=self.experiment_index)
        training.to_pickle(self.endpoint + "training.p")


class TestArrayHook(ArrayHook):
    def __call__(self):
        # Only activate during testing
        if not self.agent.training:
            super(TestArrayHook, self).__call__()

    def _run_call(self):
        # Only activate during testing
        if not self.agent.training:
            super(TestArrayHook, self)._run_call()
