from .hook import ValidationHook


class Hooks:
    """Class to instiantiate and call multiple hooks"""

    def __init__(self, agent, hooks=None):
        self.agent = agent
        # Use a validationHook by default
        self.hooks = [ValidationHook(agent)]

        if hooks is not None:
            for hook in hooks:
                self.append(hook)

    def __call__(self):
        """Call each of the hooks"""
        for hook in self.hooks:
            hook()

    def __repr__(self):
        return(str(self.hooks))

    def __iter__(self):
        return(iter(self.hooks))

    def run_end(self):
        for hook in self.hooks:
            hook._run_call()

    def append(self, hook):
        hook._register(self.agent)
        self.hooks.append(hook)


class ExperimentHooks(Hooks):
    def __init__(self, hooks=None):
        self.hooks = []
        # TODO: Register the experiment
        # self.experiment = experiment
        if hooks is not None:
            for hook in hooks:
                self.append(hook)

    def append(self, hook):
        # hook._register(self.agent)
        self.hooks.append(hook)

    def experiment_end(self):
        for hook in self.hooks:
            hook._experiment_call()


class ExperimentsHooks(Hooks):
    def __init__(self, hooks=None):
        self.hooks = []
        # TODO: Register the experiments
        # self.experiments = experiments
        if hooks is not None:
            for hook in hooks:
                self.append(hook)

    def append(self, hook):
        # hook._register(self.agent)
        self.hooks.append(hook)

    def experiment_end(self):
        for hook in self.hooks:
            hook._experiments_call()
