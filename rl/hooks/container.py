from .hook import ValidationHook


class HooksContainer(object):
    """Abstract hooks class"""
    def __iter__(self):
        return(iter(self.hooks))


class AgentHooksContainer(HooksContainer):
    """Container class to instiantiate, register, initialize and call multiple hooks"""
    def __init__(self, agent, hooks=None):
        self.agent = agent
        # Use a validationHook by default
        self.hooks = [ValidationHook(agent_id=agent.id)]

        if hooks is not None:
            for hook in hooks:
                self.append(hook)

    def step_init(self):
        for hook in self.hooks:
            hook.step_init()

    def step_end(self):
        for hook in self.hooks:
            hook.step_end()

    def __call__(self):
        """Convenience method around step_end"""
        self.step_end()

    def episode_init(self):
        for hook in self.hooks:
            hook.episode_init()

    def episode_end(self):
        for hook in self.hooks:
            hook.episode_end()

    def run_init(self):
        for hook in self.hooks:
            hook.run_init()

    def run_end(self):
        for hook in self.hooks:
            hook.run_end()

    def append(self, hook):
        # Only append hooks bound to this agent object
        if hook.agent_id == self.agent.id or hook.agent_id is "all" or (hook.agent_id == "default" and self.agent.attributes["default"]):
            # Register the agent and initiliaze the hook
            hook.register_agent(self.agent)
            hook.agent_init()
            self.hooks.append(hook)

    def __repr__(self):
        return("AgentHooks object, holding:\n" + repr(self.hooks))


class ExperimentHooksContainer(HooksContainer):
    def __init__(self, experiment, hooks=None):
        self.hooks = []
        self.experiment = experiment
        if hooks is not None:
            for hook in hooks:
                self.append(hook)

    def append(self, hook):
        # Only append hooks bound to this experiment
        if hook.experiment_id == self.experiment.id or hook.experiment_id is "all" or (hook.experiment_id == "default" and self.experiment.attributes["default"]):
            hook.register_experiment(self.experiment)
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


class ExperimentsHooksContainer(HooksContainer):
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
        hook.register_experiments(self.experiments)
        hook.experiments_init()
        self.hooks.append(hook)

    def experiments_end(self):
        for hook in self.hooks:
            hook.experiments_end()

    def __repr__(self):
        return("ExperimentsHooks object, holding:\n" + repr(self.hooks))
