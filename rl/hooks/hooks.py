from .hook import ValidationHook


class Hooks:
    """Container class to instiantiate, register, initialize and call multiple hooks"""
    def __init__(self, agent, hooks=None):
        self.agent = agent
        # Use a validationHook by default
        self.hooks = [ValidationHook(agent_id=agent.id)]

        if hooks is not None:
            for hook in hooks:
                # Only append hooks bound to this agent object
                if self.agent == hook.agent:
                    self.append(hook)

    def __call__(self):
        """Call each of the hooks"""
        for hook in self.hooks:
            hook()

    def __iter__(self):
        return(iter(self.hooks))

    def run_end(self):
        for hook in self.hooks:
            hook.run_end()

    def append(self, hook):
        if self.agent == hook.agent:
            hook.agent_init()
            self.hooks.append(hook)
        else:
            raise(Exception("Can't append: Hook's agent ID is {}, whereas current agent ID is {}".format(hook.experiment_id, self.experiment.id)))

    def __repr__(self):
        return("Hooks object, holding:\n" + repr(self.hooks))


class ExperimentHooks(Hooks):
    def __init__(self, experiment, hooks=None):
        self.hooks = []
        self.experiment = experiment
        if hooks is not None:
            for hook in hooks:
                self.append(hook)

    def append(self, hook):
        # Only append hooks bound to this experiment
        if self.experiment == hook.experiment:
            hook.experiment_init()
            self.hooks.append(hook)
        else:
            # We shouldn't have hooks defined on other experiments
            raise(Exception("Can't append: Hook's experiment ID is {}, whereas current experiment ID is {}".format(hook.experiment_id, self.experiment.id)))

    def experiment_end(self):
        for hook in self.hooks:
            hook.experiment_end()

    def __repr__(self):
        return("ExperimentHooks object, holding:\n" + repr(self.hooks))


class ExperimentsHooks(Hooks):
    def __init__(self, experiments, hooks=None):
        self.hooks = []
        # TODO: Register the experiments
        self.experiments = experiments
        if hooks is not None:
            for hook in hooks:
                # Only append hooks bound to this experiments object
                self.append(hook)

    def append(self, hook):
        """A method that takes care of registering the experiments on the hook and initialize it, by calling"""
        if self.experiments == hook.experiments:
            hook.experiments_init()
            self.hooks.append(hook)
        else:
            raise(Exception("Can't append: Hook's experiments ID is {}, whereas current experiments ID is {}".format(hook.experiments_id, self.experiments.id)))

    def experiments_end(self):
        for hook in self.hooks:
            hook.experiments_end()

    def __repr__(self):
        return("ExperimentsHooks object, holding:\n" + repr(self.hooks))
