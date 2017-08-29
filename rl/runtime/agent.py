from .experiment import Experiment
from .runtime import runtime
from rl.hooks import Hooks


class Agent(object):
    """Abstract class for an agent"""
    def __init__(self, experiment=None, id=None, hooks=None):
        if experiment is None:
            # Since we are using "default", we can overwrite it.
            self.experiment = Experiment("default", force=True)
        else:
            self.experiment = experiment

        # Get an ID
        if id is None:
            self.id = self.experiment.new_agent_id()
        else:
            self.id = id
            self.experiment.add_agent_id(id)
        # Register in the runtime
        runtime().register_agent(self)

        # Manage hooks
        # Initialize the Hooks
        if self.experiment.hooks is not None:
            # Be sure to copy the list
            hooks_list = list(self.experiment.hooks)
        else:
            hooks_list = []

        # Add user provided hooks
        if hooks is not None:
            hooks_list += hooks

        self.hooks = Hooks(self, hooks_list)

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
