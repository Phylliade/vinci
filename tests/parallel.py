from rl.experiments.parallel import ParallelExperiments


my_expe = ParallelExperiments("test", force=True, path="/tmp/experiments")


def my_script(experiment):
    # Get the environment
    # And populate it with useful metadata
    pass


with my_expe:
    my_expe.experiments(1, my_script)
