from rl.runtime.runtime import runtime


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
    * agent.policy
    * agent.goal
    * agent.achievement
    * agent.error

    :param agent: the RL agent
    :param episodic: Whether the hook will use episode information
    """
    def __init__(self, agent_id="default", experiment_id="default"):
        """
        Specify the agent object the hook must monitor
        If left to None, the hook will use the default agent
        """
        self.runtime = runtime()
        self.agent_id = agent_id
        self.experiment_id = experiment_id

    def step_init(self):
        pass

    def step_end(self):
        pass

    # Optional calls
    def episode_init(self):
        pass

    def episode_end(self):
        pass

    def run_init(self):
        pass

    def run_end(self):
        pass

    def agent_init(self):
        """Callback that is called when the agent is initialized"""
        pass

    def experiment_init(self):
        """Callback that is called when the experiment is initialized"""
        pass

    def experiment_end(self):
        pass

    def experiments_init(self):
        """Callback that is called when the experiments object is initialized"""
        pass

    def experiments_end(self):
            pass

    def register_agent(self, agent):
        self.agent = agent
        # Also register the experiment in case the hook has directly been given to the agent
        if not hasattr(self, "experiment"):
            self.register_experiment(agent.experiment)

    def register_experiment(self, experiment):
        self.experiment = experiment

    def register_experiments(self, experiments):
        self.experiments = experiments

    # @property
    # def experiments(self):
    #     return(self.runtime.get_experiments())
    #
    # @property
    # def experiment(self):
    #     return(self.runtime.get_experiment(self.experiment_id))
    #
    # @property
    # def agent(self):
    #     return(self.runtime.get_agent(self.agent_id))

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
