from .experiment import Experiment
from .runtime import runtime
from rl.hooks import Hooks


class Agent(object):
    """Abstract class for an agent"""
    def __init__(self, experiment=None, hooks=None, name=None):
        if experiment is None:
            # Since we are using "default", we can overwrite it.
            self.experiment = Experiment(experiment_id="default", force=True)
        else:
            self.experiment = experiment

        # FIXME: add_agent is dependant on the name, whereas the name (if none) requires the id...
        # Get an ID
        self.id = self.experiment.get_new_agent_id()

        # Get a name if not given
        if name is None:
            self.name = "agent-" + str(self.id)
        else:
            self.name = name

        self.experiment.add_agent(self)

        # End of initialization
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
