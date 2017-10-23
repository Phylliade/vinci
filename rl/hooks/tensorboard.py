import tensorflow as tf
from .hook import Hook


class TensorboardHook(Hook):
    def agent_init(self, *args, **kwargs):
        super(TensorboardHook, self).agent_init(*args, **kwargs)
        self.endpoint = self.experiment.endpoint("tensorboard")
        self.summary_writer = tf.summary.FileWriter(self.endpoint)

    def step_end(self):
        # Step summaries
        for summary in list(self.agent.step_summaries) + list(self.agent.step_summaries_post): 
            # FIXME: Use only one summary
            self.summary_writer.add_summary(summary, self.agent.step)

    def episode_end(self):
        # Episode summaries
        episode_summary = tf.Summary(value=[
            tf.Summary.Value(
                tag="episode_reward",
                simple_value=self.agent.episode_reward),
        ])
        self.summary_writer.add_summary(episode_summary,
                                        self.agent.episode)
