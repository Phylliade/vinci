import tensorflow as tf
from .hook import Hook


class TensorboardHook(Hook):
    def __init__(self, *args, **kwargs):
        super(TensorboardHook, self).__init__(*args, **kwargs)
        self.summary_writer = tf.summary.FileWriter('./logs')

    def __call__(self):
        # Step summaries
        for summary in list(self.agent.step_summaries):
            # FIXME: Use only one summary
            self.summary_writer.add_summary(summary, self.agent.step)

        # Episode summaries
        if self.agent.done:
            episode_summary = tf.Summary(value=[
                tf.Summary.Value(
                    tag="episode_reward",
                    simple_value=self.agent.episode_reward),
            ])
            self.summary_writer.add_summary(episode_summary,
                                            self.agent.episode)
