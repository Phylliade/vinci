import pandas as pd
from .hook import Hook


class GEPHook(Hook):
    """Collect data on an episode basis"""
    def experiment_init(self):
        self.endpoint = self.experiment.endpoint("data/collected")
        self.policies = []
        self.errors = []
        self.goals = []
        self.achievements = []
        self.rewards = []
        self.istraining = []
        self.steps = []
        self.training_steps = []

    def experiment_end(self):
        self.save()

    def episode_end(self):
        # Collect statistics
        self.steps.append(self.agent.step)
        self.training_steps.append(self.agent.training_step)
        self.errors.append(self.agent.error)
        self.policies.append(self.agent.policy)
        self.goals.append(self.agent.goal)
        self.achievements.append(self.agent.achievement)
        self.istraining.append(self.agent.training)
        self.rewards.append(self.agent.episode_reward)

    def save(self):
        print("Saving collected data")
        errors = pd.Series(self.errors, name="errors", index=self.steps)
        rewards = pd.Series(self.rewards, name="rewards", index=self.steps)
        goals = pd.Series(self.goals, name="goals", index=self.steps)
        istraining = pd.Series(self.istraining, name="training", index=self.steps)
        steps = pd.Series(self.steps, name="steps")
        training_steps = pd.Series(self.training_steps, name="training_steps", index=self.steps)
        df = pd.DataFrame({"errors": errors, "rewards": rewards, "goals": goals, "training": istraining, "training_steps": training_steps}, index=steps)
        df.to_pickle(self.endpoint + "data.p")


class RunGEPHook(GEPHook):
    def run_init(self):
        self.run_errors = []

    def run_end(self):
        self.errors.append(self.run_errors)
        self.run_errors = []
        self.steps.append(self.agent.step)
        self.training_steps.append(self.agent.training_step)
        self.policies.append(self.agent.policy)
        self.goals.append(self.agent.goal)
        self.achievements.append(self.agent.achievement)
        self.istraining.append(self.agent.training)
        self.rewards.append(self.agent.episode_reward)

    def episode_end(self):
        self.run_errors.append(self.agent.error)
