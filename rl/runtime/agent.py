from .experiment import DefaultExperiment
from .runtime import runtime
from rl.hooks.container import AgentHooksContainer


class Agent(object):
    """Abstract class for an agent"""
    def __init__(self, experiment, hooks=None, name=None):
        # Dict to store useful agent attributes
        self.attributes = {"default": True}

        if experiment is None:
            # Create the experiment ourselves but we are losing the possibility to use it as a context manager
            # self.experiment = DefaultExperiment()
            raise(ValueError("experiment is not provided, please provide an experiment"))
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

        # Setup hook variables
        self._hook_variables = ["training", "step", "episode", "episode_step", "done", "step_summaries"]
        self._hook_variables_optional = ["reward", "episode_reward", "observation"]
        # Set them to none as default, only if not defined
        for variable in (self._hook_variables + self._hook_variables_optional):
            setattr(self, variable, getattr(self, variable, None))

        # Persistent values
        self.step = 0
        self.training_step = 0
        self.episode = 0
        self.training_episode = 0
        self.run_number = 0
        # TODO: Use these variables
        self.environment_step = 0
        self.environment_episode = 0

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

        self.hooks = AgentHooksContainer(self, hooks_list)

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

    @property
    def session(self):
        return(self.experiment.session)
