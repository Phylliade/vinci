import os
import shutil
from rl.hooks import ExperimentHooksContainer
# from contextlib import contextmanager
from .run import Run
from ..utils.printer import print_info, print_warning
from .runtime import runtime


class PersistentExperiment(object):
    """An abstract experiment only assuming filesystem related functions"""

    def __init__(self, experiment_id, path="./experiments/", force=False):
        self.id = str(experiment_id)
        self.experiment_base = path.rstrip("/") + "/" + self.id + "/"

        # Check of the experiment dir already exists
        if os.path.exists(self.experiment_base):
            if not force:
                raise (FileExistsError(
                    "The directory {} already exists, remove it or use the `force` argument".
                    format(self.experiment_base)))
            else:
                print_warning("Overwriting {}".format(self.experiment_base))
                shutil.rmtree(self.experiment_base)
        os.makedirs(self.experiment_base)
        self.endpoints = []

        # Dict to store useful agent attributes
        self.attributes = {"default": True}

        # No default experiment by default
        # It will be configured later
        self.has_default_experiment = False

    def endpoint(self, path):
        full_path = self.experiment_base + path.rstrip("/") + "/"
        # An endpoint could already exist, because used elsewhere in the experiment
        if (path not in self.endpoints):
            self.endpoints.append(path)
            if not os.path.exists(full_path):
                # Remove use of exist_ok for py2
                os.makedirs(full_path)
        return (full_path)


class Experiment(PersistentExperiment):
    def __init__(self, experiment_id, experiments=None, hooks=None, use_tf=True, tf_config=None, analytics=False, seed=None, **kwargs):
        super(Experiment, self).__init__(experiment_id, **kwargs)

        self.count = 1
        self.done = False
        self.experiments = experiments
        self.run_count = 0
        self.next_agent_id = 0
        # We only need to store the agent names, not the agent themselves for the moment
        self.agents = {}

        # End of init
        # Register in the runtime
        runtime().register_experiment(self)
        # Add the hooks
        if self.experiments is not None:
            if self.experiments.hooks is not None:
                # Be sure to copy the list
                hooks_list = list(self.experiments.hooks)
        else:
            hooks_list = []

        # Experiment-level hooks
        if analytics:
            from rl.hooks.arrays import ExperimentArrayHook
            hooks_list.append(ExperimentArrayHook())

        # Add user-provided hooks
        if hooks is not None:
            hooks_list += hooks

        self.hooks = ExperimentHooksContainer(self, hooks=hooks_list)

        if use_tf:
            import tensorflow as tf
            from keras import backend as K
            # Use Keras's sessions
            # self.session = K.get_session()
            self.session = tf.Session(config=tf_config)
            K.set_session(self.session)

        # Seed
        if seed is not None:
            import random
            import numpy.random
            print_info("Using seed {}".format(seed))
            random.seed(seed)
            numpy.random.seed(seed)
            if use_tf:
                tf.set_random_seed(seed)

    def __enter__(self):
        print_info("Beginning experiment {}".format(self.id))

    def __exit__(self, *args):
        self.done = True
        self.hooks.experiment_end()

    def __next__(self):
        run = Run()
        self.run_count += 1
        return (run)

    def get_new_agent_id(self):
        return (self.next_agent_id)

    def add_agent(self, agent):
        """Add the agent to the experiment registers, and return its id"""
        if agent.name in self.agents:
            raise (NameError(
                "An agent with the name {} already exists".format(agent.name)))
        else:
            self.agents[agent.name] = agent
            self.next_agent_id += 1

            # Manage the default agent
            if not self.has_default_experiment:
                agent.attributes["default"] = True
                self.has_default_experiment = True
            else:
                agent.attributes["default"] = False


class DefaultExperiment(Experiment):
    """
    An experiment used by default by the agents.
    If will use the id "default" and always overwrite an existing experiment
    """

    def __init__(self, *args, **kwargs):
        super(DefaultExperiment, self).__init__(
            *args, experiment_id="default", force=True, **kwargs)


class RootExperiment(PersistentExperiment):
    """A special experiment associated with an experiments object"""

    def __init__(self, experiments, **kwargs):
        super(RootExperiment, self).__init__(
            experiment_id=experiments.name, path=experiments._path, **kwargs)
