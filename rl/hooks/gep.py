import numpy as np
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
        self.istraining = []
        self.step_index = []

    def experiment_end(self):
        self.save()

    def episode_end(self):
        # Collect statistics
        self.step_index.append(self.agent.step)
        self.errors.append(self.agent.error)
        self.policies.append(self.agent.policy)
        self.goals.append(self.agent.goal)
        self.achievements.append(self.agent.achievement)
        self.istraining.append(self.agent.training)

    def save(self):
        print("Saving collected data")
        errors = pd.Series(self.errors, name="errors", index=self.step_index)
        errors.to_pickle(self.endpoint + "step.p")
