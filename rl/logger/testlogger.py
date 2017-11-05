from rl.logger.callbacks import Callback

class TestLogger(Callback):
    def on_train_begin(self, logs):
        print('Testing for {} episodes ...'.format(self.params['episodes']))

    def on_episode_end(self, episode, logs):
        template = 'Episode {0}: reward: {1:.3f}, steps: {2}'
        variables = [
            episode,
            logs['episode_reward'],
            logs['steps'],
        ]
        print(template.format(*variables))
