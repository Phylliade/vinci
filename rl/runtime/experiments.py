from .experiment import Experiment
from ..hooks import ExperimentsHooks


class Experiments():
    def __init__(self, name, analytics=True, hooks=None, force=False, path="./experiments"):
        self.name = str(name)
        self.done = False
        self._path = path
        # A special experiment providing endpoints for the experiments object itself
        self._root_experiment = Experiment(self.name, experiments=self, force=force, path=self._path)
        if hooks is None:
            hooks = []
        if analytics:
            from rl.hooks.arrays import ArrayHook
            hooks.append(ArrayHook())
        self.hooks = ExperimentsHooks(self, hooks=hooks)

    def endpoint(self, path):
        return(self._root_experiment.endpoint(path))

    def __call__(self, number):
            for epoch in range(1, number + 1):
                print("Beginning experiment {}".format(epoch))
                experiment = Experiment(self.name + "/" + str(epoch), experiments=self, hooks=self.hooks, path=self._path)

                yield experiment

                experiment.done = True
                experiment.hooks.experiment_end()

                if (epoch) == number:
                    self.done = True
                    self.hooks.experiments_end()
