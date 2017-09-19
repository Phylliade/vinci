from .multiple import MultipleExperiments


class SequentialExperiments(MultipleExperiments):
    def __init__(self, name, analytics=False, hooks=None, **kwargs):
        # Add the hooks
        if hooks is None:
            hooks = []

        # Add analytics hooks
        if analytics:
            from rl.hooks.arrays import ArrayHook
            hooks.append(ArrayHook())

        super(SequentialExperiments, self).__init__(name, hooks=hooks, **kwargs)
