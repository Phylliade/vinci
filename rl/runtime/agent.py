from .experiment import Experiment


class Agent(object):
    """Abstract class for an agent"""
    def __init__(self, experiment=None):
        if experiment is None:
            # Since we are using "default", we can overwrite it.
            self.experiment = Experiment("default", force=True)
        else:
            self.experiment = experiment

    def _run(self, train=True):
        raise(NotImplementedError())

    def train(self, **kwargs):
        self.run(train=True, **kwargs)

    def test(self, **kwargs):
        self.run(train=False, **kwargs)
