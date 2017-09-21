from .hook import Hook
import pandas as pd


class ArrayHook(Hook):
    """Collect data during the span of multiple experiments, run by run"""
    def experiments_init(self):
        # An experiments-wide endpoint
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

    def experiments_end(self):
        pass
        # self.save()

    def experiment_init(self):
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

    def experiment_end(self):
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

        # Save every experiment
        if (self.experiments.experiment_count % 1 == 0):
            self.save()

    # TODO: Remove this
    def agent_init(self):
        self.run_rewards = []

    # TODO: Move this to episode_end method
    def step_end(self):
        if self.agent.done:
            self.run_rewards.append(self.agent.episode_reward)

    def run_init(self):
        self.run_rewards = []

    def run_end(self):
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


class ExperimentArrayHook(Hook):
    """Collect data during the span of an experiment, run by run"""
    def experiment_init(self):
        self.endpoint = self.experiment.endpoint("data")

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

    def experiment_end(self):
        self.save()

    # TODO: Remove this
    def agent_init(self):
        self.run_rewards = []

    # TODO: Move this to episode_end method
    def step_end(self):
        if self.agent.done:
            self.run_rewards.append(self.agent.episode_reward)

    def run_init(self):
        self.run_rewards = []

    def run_end(self):
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

    def save(self):
        print("Saving data")
        data = pd.DataFrame({"rewards": self.experiment_rewards, "episode": self.experiment_episode, "is_training": self.experiment_istraining, "step": self.experiment_step}, index=self.run_index)
        data.to_pickle(self.endpoint + "data.p")
