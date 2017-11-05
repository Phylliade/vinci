from rl.logger.callbacks import Callback
import timeit
import warnings
import numpy as np

class TrainEpisodeLogger(Callback):
    """Display information at the episode scale"""
    def __init__(self):
        # Some algorithms compute multiple episodes at once since they are multi-threaded.
        # We therefore use a dictionary that is indexed by the episode to separate episodes
        # from each other.
        self.episode_start = {}
        self.observations = {}
        self.rewards = {}
        self.actions = {}
        self.metrics = {}
        self.step = 0

    def on_train_begin(self, logs):
        self.train_start = timeit.default_timer()
        self.metrics_names = self.model.metrics_names
        print('Training for {} steps ...'.format(self.params['steps']))

    def on_train_end(self, logs):
        duration = timeit.default_timer() - self.train_start
        print('done, took {:.3f} seconds'.format(duration))

    def on_episode_begin(self, episode, logs):
        self.episode_start[episode] = timeit.default_timer()
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []
        self.metrics[episode] = []
        print("Begin episode {}".format(episode))

    def on_episode_end(self, episode, logs):
        duration = timeit.default_timer() - self.episode_start[episode]
        episode_steps = len(self.observations[episode])

        # Format all metrics.
        metrics = np.array(self.metrics[episode])
        metrics_template = ''
        metrics_variables = []
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            for idx, name in enumerate(self.metrics_names):
                if idx > 0:
                    metrics_template += ', '
                try:
                    value = np.nanmean(metrics[:, idx])
                    metrics_template += '{}: {:f}'
                except Warning:
                    value = '--'
                    metrics_template += '{}: {}'
                metrics_variables += [name, value]
        metrics_text = metrics_template.format(*metrics_variables)

        nb_step_digits = str(
            int(np.ceil(np.log10(self.params['steps']))))
        template = "End \033[32mepisode {episode}\033[0m,\n  duration: {duration:.3f}s, "
        template += "\n  total steps: {step: " + nb_step_digits + "d}/{steps}: "
        template += "\n  episode steps: {episode_steps},\n  steps per second: {sps:.0f}, "
        template += "\n  \033[32mepisode reward: {episode_reward:.3f}\033[0m,\n  mean reward: {reward_mean:.3f} [{reward_min:.3f}, {reward_max:.3f}], "
        template += "\n  mean action: {action_mean:.3f} [{action_min:.3f}, {action_max:.3f}],\n  mean observation: {obs_mean:.3f} [{obs_min:.3f}, {obs_max:.3f}], {metrics}\n---\n"
        variables = {
            'step': self.step,
            'steps': self.params['steps'],
            'episode': episode,
            'duration': duration,
            'episode_steps': episode_steps,
            'sps': float(episode_steps) / duration,
            'episode_reward': np.sum(self.rewards[episode]),
            'reward_mean': np.mean(self.rewards[episode]),
            'reward_min': np.min(self.rewards[episode]),
            'reward_max': np.max(self.rewards[episode]),
            'action_mean': np.mean(self.actions[episode]),
            'action_min': np.min(self.actions[episode]),
            'action_max': np.max(self.actions[episode]),
            'obs_mean': np.mean(self.observations[episode]),
            'obs_min': np.min(self.observations[episode]),
            'obs_max': np.max(self.observations[episode]),
            'metrics': metrics_text,
        }
        print(template.format(**variables))

        # Free up resources.
        del self.episode_start[episode]
        del self.observations[episode]
        del self.rewards[episode]
        del self.actions[episode]
        del self.metrics[episode]

    def on_step_end(self, step, logs):
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])
        self.metrics[episode].append(logs['metrics'])
        self.step += 1
