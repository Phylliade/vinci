from .hook import Hook
import pandas as pd


class ArrayHook(Hook):
    def _register_experiments(self, *args, **kwargs):
        super(ArrayHook, self)._register_experiments(*args, **kwargs)
        self.endpoint = self.experiments.endpoint("data")

        # Rewards
        self.experiments_rewards = []

        # Episodes
        self.experiments_episodes = []

        # Training
        self.experiments_istraining = []

        # Experiment index
        # Lists of experiments
        self.experiment_index = []

    def _register_experiment(self, *args, **kwargs):
        super(ArrayHook, self)._register_experiment(*args, **kwargs)

        # Rewards
        self.experiment_rewards = []

        # Episodes
        self.experiment_episode = []

        # Training
        self.experiment_istraining = []

    # TODO: Put run_rewards in _register_run
    def _register(self, *args, **kwargs):
        super(ArrayHook, self)._register(*args, **kwargs)
        self.run_rewards = []

    def __call__(self):
        if self.agent.done:
            self.run_rewards.append(self.agent.episode_reward)

    def _run_call(self):
        # Rewards
        self.experiment_rewards.append(self.run_rewards)
        self.run_rewards = []
        # Episodes
        self.experiment_episode.append(self.agent.episode)
        # IsTraining
        self.experiment_istraining.append(self.agent.training)

    def _experiment_call(self):
        # Rewards
        self.experiments_rewards.append(self.experiment_rewards)
        self.experiment_rewards = []

        # IsTraining
        self.experiments_istraining.append(self.experiment_istraining)
        self.experiment_istraining = []

        # Episodes count
        self.experiments_episodes.append(self.experiment_episode)
        self.experiment_episode = []

        # Experiment index
        self.experiment_index.append(self.experiments.experiment_count)

        # Save every 5 experiments
        if (self.experiments.experiment_count % 5 == 0):
            self.save()

    def _experiments_call(self):
        self.save()

    def save(self):
        print("Saving data")
        # each of the arrays' cells store data for one run
        rewards = pd.DataFrame(self.experiments_rewards, index=self.experiment_index)
        rewards.columns.name = "runs"
        rewards.index.name = "experiment"
        rewards.to_pickle(self.endpoint + "reward.p")

        episodes = pd.DataFrame(self.experiments_episodes, index=self.experiment_index)
        episodes.to_pickle(self.endpoint + "episodes.p")

        training = pd.DataFrame(self.experiments_istraining, index=self.experiment_index)
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
