DEFAULT_AGENT = 0


class Runtime(object):
    """A object keeping a trace of the whole runtime hierarchy"""
    # TODO: Remove the unused agents / experiments / experiments objects...
    # And add a minimal GC to avoid leaks
    def __init__(self):
        self.agents = {}
        self.default_agent = None
        self.current_experiment = None
        self.current_experiments = None

    def register_agent(self, agent):
        # Overwrite agent if it's already existing
        if agent.id in self.agents:
            del self.agents[agent.id]
        self.agents[agent.id] = agent

    def register_experiment(self, experiment):
        del self.current_experiment
        self.current_experiment = experiment

    def register_experiments(self, experiments):
        del self.current_experiments
        self.current_experiments = experiments

    def get_agent(self, id=None):
        if id is None:
            agent = self.agents[DEFAULT_AGENT]
        else:
            agent = self.agents[id]

        return(agent)

    def get_experiment(self):
        return(self.current_experiment)

    def get_experiments(self, id=None):
        return(self.current_experiments)


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
