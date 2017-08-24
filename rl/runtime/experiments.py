from .experiment import Experiment
from ..hooks import ExperimentsHooks
from ..utils.printer import print_info


class Experiments():
    def __init__(self, name, analytics=False, hooks=None, force=False, path="./experiments"):
        self.name = str(name)
        self.done = False
        self.experiment_count = 0
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

    def experiments(self, number):
        print_info("Beginning {} experiments".format(number))

        for epoch in range(1, number + 1):
            print_info("Experiment {}/{}")
            experiment = Experiment(self.name + "/" + str(epoch), experiments=self, hooks=self.hooks, path=self._path)
            self.experiment_count += 1
            with experiment:
                yield experiment

    def __enter__(self):
        print_info("Begin experiments under prefix {}".format(self.name))

    def __exit__(self, *args):
        self.done = True
        self.hooks.experiments_end()
