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

    def __init__(self, agent=None, episodic=True):
        """

        """
        self.agent = agent
        self.episodic = episodic

    def __call__(self):
        raise (NotImplementedError)

    def _register(self, agent):
        """Register the agent"""
        self.agent = agent

    @property
    def count(self):
        return (self.agent.episode)


class ValidationHook(Hook):
    """Perform validation of the hooks variables at runtime"""

    def __init__(self, *args, **kwargs):
        super(ValidationHook, self).__init__(*args, **kwargs)
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


class Hooks:
    """Class to instiantiate and call multiple hooks"""

    def __init__(self, agent, hooks=None):
        self.agent = agent
        # Use a validationHook by default
        self.hooks = [ValidationHook(agent)]

        if hooks is not None:
            for hook in hooks:
                hook._register(agent=agent)
                self.hooks.append(hook)

    def __call__(self):
        """Call each of the hooks"""
        for hook in self.hooks:
            hook()

    def append(self, hook):
        hook.register(self.agent)
        self.hooks.append(hook)
