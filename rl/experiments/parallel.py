import multiprocessing

from .multiple import MultipleExperiments
from rl.runtime.experiment import Experiment
from rl.utils.printer import print_info
from rl.hooks.arrays import ExperimentArrayHook


class ParallelExperiments(MultipleExperiments):
    def experiments(self, number, script_function, nb_processes=16):
        """Execute the given function in different subprocesses"""
        print_info("Beginning {} experiments".format(number))

        for experiment_count in range(1, number + 1):
            print_info("Spinning experiment {}/{}".format(experiment_count, number))
            experiment_id = str(experiment_count)
            experiment_id_full = (self.name + "/" + experiment_id)

            experiment = Experiment(experiment_id=experiment_id_full, experiments=self, path=self._path, hooks=[ExperimentArrayHook()])

            # Run the experiment in a separate process
            experiment_process = multiprocessing.Process(name=experiment_id, target=experiment_function, args=(script_function, experiment))
            experiment_process.start()


def experiment_function(script_function, experiment):
    with experiment:
        script_function(experiment)
