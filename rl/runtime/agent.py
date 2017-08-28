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
        """
        Train the agent. On the contrary of :func:`test`, learning is involved
        See :func:`_run` for the argument list.
        """
        return(self._run(train=True, **kwargs))

    def test(self, **kwargs):
        """
        Test the agent. On the contrary of :func:`fit`, no learning is involved
        See :func:`_run` for the argument list.
        """
        return(self._run(train=False, **kwargs))
