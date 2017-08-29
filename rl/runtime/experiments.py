from .experiment import Experiment, RootExperiment
from ..hooks import ExperimentsHooks
from ..utils.printer import print_info
from .runtime import runtime


class Experiments(object):
    def __init__(self, name, analytics=False, hooks=None, force=False, path="./experiments"):
        self.name = str(name)
        self.id = name

        self.done = False
        self.experiment_count = 0
        self._path = path
        # A special experiment providing endpoints for the experiments object itself
        self._root_experiment = RootExperiment(experiments=self, force=force)

        # End of init
        # Register in the runtime
        runtime().register_experiments(self)

        # Add the hooks
        if hooks is None:
            hooks = []
        if analytics:
            from rl.hooks.arrays import ArrayHook
            hooks.append(ArrayHook())
        self.hooks = ExperimentsHooks(self, hooks=hooks)

    def endpoint(self, path):
        return(self._root_experiment.endpoint(path))

    def experiments(self, number):
        """Get an iterable that returns sequential experiments"""
        print_info("Beginning {} experiments".format(number))

        for _ in range(1, number + 1):
            self.experiment_count += 1
            print_info("Experiment {}/{}".format(self.experiment_count, number))
            experiment = Experiment(experiment_id=self.name + "/" + str(self.experiment_count), experiments=self, path=self._path)
            with experiment:
                yield experiment

    def __enter__(self):
        print_info("Begin experiments under prefix {}".format(self.name))

    def __exit__(self, *args):
        self.done = True
        self.hooks.experiments_end()
