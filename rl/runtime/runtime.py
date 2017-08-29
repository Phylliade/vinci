class Runtime(object):
    """A object keeping a trace of the whole runtime hierarchy"""
    def __init__(self):
        self.agents = {}
        self.default_agent = None
        self.experiment = {}
        self.default_experiment = None
        self.experiments = {}
        self.default_experiments = None

    def register_agent(self, agent):
        self.agents[agent.id] = agent
        if self.default_agent is None:
            self.default_agent = agent

    def register_experiment(self, experiment):
        self.experiment[experiment.id] = experiment
        if self.default_experiment is None:
            self.default_experiment = experiment

    def register_experiments(self, experiments):
        self.experiment[experiments.id] = experiments
        if self.default_experiments is None:
            self.default_experiments = experiments

    def get_agent(self, id=None):
        if id is None:
            agent = self.default_agent
        else:
            agent = self.agents[id]

        return(agent)

    def get_experiment(self, id=None):
        if id is None:
            experiment = self.default_experiment
        else:
            experiment = self.experiment[id]

        return(experiment)

    def get_experiments(self, id=None):
        if id is None:
            experiments = self.default_experiments
        else:
            experiments = self.experiments[id]

        return(experiments)

# Global runtime variable
_runtime = Runtime()


def runtime():
    return(_runtime)


def run(self, epochs):
    self.training = True
    self.done = True

    # We could use a termination criterion, based on step instead of epoch, as in  _run
    # TODO: Trigger an exception
    for epoch in range(epochs):
        if self.done:
            self.episode += 1
            self.episode_reward = 0.
            self.episode_step = 0

        # Initialize the step
        self.done = False
        self.step += 1
        self.episode_step += 1
        self.step_summaries = []

        # Run the step
        yield epoch

        # Close the step
