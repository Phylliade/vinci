from keras.utils.generic_utils import Progbar
from rl.logger.callbacks import Callback
import timeit
import numpy as np

class TrainIntervalLogger(Callback):
    def __init__(self, interval=10000):
        self.interval = interval
        self.step = 0
        self.reset()

    def reset(self):
        self.interval_start = timeit.default_timer()
        self.progbar = Progbar(target=self.interval)
        self.metrics = []
        self.infos = []
        self.info_names = None
        self.episode_rewards = []

    def on_train_begin(self, logs):
        self.train_start = timeit.default_timer()
        self.metrics_names = self.model.metrics_names
        print('Training for {} steps ...'.format(self.interval))

    def on_train_end(self, logs):
        duration = timeit.default_timer() - self.train_start
        print(' done, took {:.3f} seconds'.format(duration))

    def on_step_begin(self, step, logs):
        if self.step % self.interval == 0:
            if len(self.episode_rewards) > 0:
                metrics = np.array(self.metrics)
                assert metrics.shape == (self.interval,len(self.metrics_names))
                formatted_metrics = ''
                if not np.isnan(metrics).all():  # not all values are means
                    means = np.nanmean(self.metrics, axis=0)
                    assert means.shape == (len(self.metrics_names), )
                    for name, mean in zip(self.metrics_names, means):
                        formatted_metrics += ' - {}: {:.3f}'.format(name, mean)

                formatted_infos = ''
                if len(self.infos) > 0:
                    infos = np.array(self.infos)
                    if not np.isnan(infos).all():  # not all values are means
                        means = np.nanmean(self.infos, axis=0)
                        assert means.shape == (len(self.info_names), )
                        for name, mean in zip(self.info_names, means):
                            formatted_infos += ' - {}: {:.3f}'.format(
                                name, mean)
                print(
                    '{} episodes - episode_reward: {:.3f} [{:.3f}, {:.3f}]{}{}'.
                    format(
                        len(self.episode_rewards),
                        np.mean(self.episode_rewards),
                        np.min(self.episode_rewards),
                        np.max(self.episode_rewards), formatted_metrics,
                        formatted_infos))
                print('')
            self.reset()
            print('Interval {} ({} steps performed)'.format(self.step // self.interval, self.step))

    def on_step_end(self, step, logs):
        if self.info_names is None:
            self.info_names = logs['info'].keys()
        values = [('reward', logs['reward'])]
        self.progbar.update(
            (self.step % self.interval), values=values, force=True)
        self.step += 1
        self.metrics.append(logs['metrics'])
        if len(self.info_names) > 0:
            self.infos.append([logs['info'][k] for k in self.info_names])

    def on_episode_end(self, episode, logs):
        self.episode_rewards.append(logs['episode_reward'])


class ModelIntervalCheckpoint(Callback):
    def __init__(self, filepath, interval, verbosity=0):
        super(ModelIntervalCheckpoint, self).__init__()
        self.filepath = filepath
        self.interval = interval
        self.verbosity = verbosity
        self.total_steps = 0

    def on_step_end(self, step, logs={}):
        self.total_steps += 1
        if self.total_steps % self.interval != 0:
            # Nothing to do.
            return

        filepath = self.filepath.format(step=self.total_steps, **logs)
        if self.verbosity > 0:
            print('Step {}: saving model to {}'.format(self.total_steps, filepath))
        self.model.save_weights(filepath, overwrite=True)
