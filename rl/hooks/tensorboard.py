import tensorflow as tf
from .hook import Hook


class TensorboardHook(Hook):
    def _agent_init(self, *args, **kwargs):
        super(TensorboardHook, self)._agent_init(*args, **kwargs)
        self.endpoint = self.experiment.endpoint("tensorboard")
        self.summary_writer = tf.summary.FileWriter(self.endpoint)

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
