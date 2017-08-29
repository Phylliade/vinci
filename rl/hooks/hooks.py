from .hook import ValidationHook


class Hooks:
    """Class to instiantiate and call multiple hooks"""
    def __init__(self, agent, hooks=None):
        self.agent = agent
        # Use a validationHook by default
        self.hooks = [ValidationHook(agent_id=agent.id)]

        if hooks is not None:
            for hook in hooks:
                # Only append hooks bound to this agent object
                if self.agent.id == hook.agent_id:
                    self.append(hook)
                    hook._agent_init()

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
        # if self.agent.id == hook.agent_id:
        #     hook.register_(agent)

        # hook._register()
        self.hooks.append(hook)


class ExperimentHooks(Hooks):
    def __init__(self, experiment, hooks=None):
        self.hooks = []
        self.experiment = experiment
        if hooks is not None:
            for hook in hooks:
                # Only append hooks bound to this experiment
                if self.experiment.id == hook.experiment_id:
                    self.append(hook)
                    hook._experiment_init()

    def append(self, hook):
        # hook._register_experiment(self.experiment)
        self.hooks.append(hook)

    def experiment_end(self):
        for hook in self.hooks:
            hook._experiment_call()


class ExperimentsHooks(Hooks):
    def __init__(self, experiments, hooks=None):
        self.hooks = []
        # TODO: Register the experiments
        self.experiments = experiments
        if hooks is not None:
            for hook in hooks:
                # Only append hooks bound to this experiments object
                if self.experiments.id == hook.experiments_id:
                    self.append(hook)
                    hook._experiments_init()

    def append(self, hook):
        # hook._register_experiments(self.experiments)
        self.hooks.append(hook)

    def experiments_end(self):
        for hook in self.hooks:
            hook._experiments_call()
