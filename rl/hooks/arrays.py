from .hook import Hook
import pandas as pd


class ArrayHook(Hook):
    def __init__(self, *args, **kwargs):
        super(ArrayHook, self).__init__(*args, **kwargs)
        self.endpoint = self.experiments.endpoint("data")

    def _experiments_init(self):
        self.endpoint = self.experiments.endpoint("data")

        # Rewards
        self.experiments_rewards = []

        # Episode
        self.experiments_episode = []

        # Step
        self.experiments_step = []

        # Training
        self.experiments_istraining = []

        # Indexes
        # TODO: Precompute these values
        # Lists of experiments
        self.experiment_index = []

    def _experiment_init(self):
        # Rewards
        self.experiment_rewards = []

        # Episode
        self.experiment_episode = []

        # Step
        self.experiment_step = []

        # Training
        self.experiment_istraining = []

        # Run index
        self.run_index = []

    # TODO: Put run_rewards in _register_run
    def _agent_init(self):
        self.run_rewards = []

    def __call__(self):
        if self.agent.done:
            self.run_rewards.append(self.agent.episode_reward)

    def _run_call(self):
        # Rewards
        self.experiment_rewards.append(self.run_rewards)
        self.run_rewards = []
        # Episode
        self.experiment_episode.append(self.agent.episode)
        # IsTraining
        self.experiment_istraining.append(self.agent.training)
        # Step
        self.experiment_step.append(self.agent.training_step)

        # Run index
        self.run_index.append(self.agent.run_number)

    def _experiment_call(self):
        # Rewards
        self.experiments_rewards.append(self.experiment_rewards)
        self.experiment_rewards = []

        # IsTraining
        self.experiments_istraining.append(self.experiment_istraining)
        self.experiment_istraining = []

        # Episode count
        self.experiments_episode.append(self.experiment_episode)
        self.experiment_episode = []

        # Step count
        self.experiments_step.append(self.experiment_step)
        self.experiment_step = []

        # Experiment index
        self.experiment_index.append(self.experiments.experiment_count)

        # Save every at experiment
        if (self.experiments.experiment_count % 1 == 0):
            self.save()

    def _experiments_call(self):
        pass
        # self.save()

    def save(self):
        print("Saving data")
        # each of the arrays' cells store data for one run
        rewards = pd.DataFrame(self.experiments_rewards, index=self.experiment_index, columns=self.run_index)
        rewards.columns.name = "runs"
        rewards.index.name = "experiment"
        rewards.to_pickle(self.endpoint + "reward.p")

        episode = pd.DataFrame(self.experiments_episode, index=self.experiment_index, columns=self.run_index)
        episode.columns.name = "runs"
        episode.index.name = "experiment"
        episode.to_pickle(self.endpoint + "episode.p")

        training = pd.DataFrame(self.experiments_istraining, index=self.experiment_index, columns=self.run_index)
        training.columns.name = "runs"
        training.index.name = "experiment"
        training.to_pickle(self.endpoint + "training.p")

        step = pd.DataFrame(self.experiments_step, index=self.experiment_index, columns=self.run_index)
        step.columns.name = "runs"
        step.index.name = "experiment"
        step.to_pickle(self.endpoint + "step.p")


class TestArrayHook(ArrayHook):
    def __call__(self):
        # Only activate during testing
        if not self.agent.training:
            super(TestArrayHook, self).__call__()

    def _run_call(self):
        # Only activate during testing
        if not self.agent.training:
            super(TestArrayHook, self)._run_call()
