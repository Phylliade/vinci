import os
import shutil
from rl.hooks import ExperimentHooks


class Experiment(object):
    def __init__(self, experiment_id, path="./experiments/", force=False, experiments=None, hooks=None):
        self.id = str(experiment_id)
        self.experiment_base = path.rstrip("/") + "/" + self.id + "/"
        self.count = 1
        self.done = False
        self.experiments = experiments
        self.hooks = ExperimentHooks(self, hooks=hooks)

        # Check of the experiment dir already exists
        if os.path.exists(self.experiment_base):
            if not force:
                raise(FileExistsError("The directory {} already exists, remove it or use the `force` argument".format(self.experiment_base)))
            else:
                print("Overwriting {}".format(self.experiment_base))
                shutil.rmtree(self.experiment_base)
        os.makedirs(self.experiment_base)
        self.endpoints = []

    def endpoint(self, path):
        self.endpoints.append(path)
        full_path = self.experiment_base + path.rstrip("/") + "/"
        # At this point, the experiment path should already be cleared, so there is no need to check for existence of the file
        os.makedirs(full_path, exist_ok=True)
        return(full_path)
