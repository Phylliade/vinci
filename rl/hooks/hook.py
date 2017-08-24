class Hook(object):
    """
    The abstract Hook class.
    A hook is designed to be a callable running on an agent object. It shouldn't return anything and instead exports the data itself (e.g. pickle, image).
    It is run at the end of **each step**.

    The hook API relies on the following agent attributes, always available:

    * agent.training: boolean: Whether the agent is in training mode
    * agent.step: int: the step number. Begins to 1.
    * agent.reward: The reward of the current step
    * agent.episode: int: The current episode. Begins to 1.
    * agent.episode_step: int: The step count in the current episode. Begins to 1.
    * agent.done: Whether the episode is terminated
    * agent.step_summaries: A list of summaries of the current step

    These variables may also be available:
    * agent.episode_reward: The cumulated reward of the current episode
    * agent.observation: The observation at the beginning of the step
    * agent.observation_1: The observation at the end of the step
    * agent.action: The action taken during the step

    :param agent: the RL agent
    :param episodic: Whether the hook will use episode information
    """
    def __init__(self, agent=None):
        if agent is not None:
            self._register(agent)
        self.registered = False

    def __call__(self):
        raise (NotImplementedError)

    # Optional calls
    def _run_call(self):
        pass

    def _experiment_call(self):
        pass

    def experiments_call(self):
        pass

    def _register_run(self, run):
        """Register the agent"""
        self.run = run
        # self.registered = True

    def _register(self, agent):
        """Register the agent"""
        self.agent = agent
        # TODO: Create a _register_experiment method
        self.experiment = agent.experiment
        self.registered = True

    def _register_experiment(self, experiment):
        """Register the agent"""
        self.experiment = experiment
        # self.registered = True

    def _register_experiments(self, experiments):
        """Register the agent"""
        self.experiments = experiments
        # self.registered = True

    @property
    def count(self):
        return (self.agent.episode)


class ValidationHook(Hook):
    """Perform validation of the hooks variables at runtime"""
    def _register(self, *args, **kwargs):
        super(ValidationHook, self)._register(*args, **kwargs)
        self.validated = False

    def __call__(self):
        # Only run this hook once
        if not self.validated:
            # Check that every hook variables are defined
            for variable in self.agent._hook_variables:
                if getattr(self.agent, variable) is None:
                    raise (ValueError(
                        "The variable {} is None, whereas it should be defined for the hook API".
                        format(variable)))
            # The test passed
            self.validated = True
