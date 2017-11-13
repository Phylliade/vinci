from rl.agents.ddpg import DDPGAgent
import tensorflow as tf

# Whether to use Keras inference engine
USE_KERAS_INFERENCE = False

import keras.backend as K


class OPDDPGAgent(DDPGAgent):
    def __init__(self, bootstrap_actor, **kwargs):

        self.bootstrap_actor = bootstrap_actor
        super(OPDDPGAgent, self).__init__(**kwargs)

    def train_critic(self, batch, sgd_iterations=1):
        """Fit the critic network"""
        # Get the target action
        # \pi_bootstrap(s_{t + 1})
        target_actions = self.session.run(
            self.bootstrap_actor(self.variables["state"]),
            feed_dict={self.variables["state"]: batch.state1,
                       K.learning_phase(): 0})
        assert target_actions.shape == (self.batch_size, self.nb_actions)

        # Get the target Q value of the next state
        # Q(s_{t + 1}, \pi(s_{t + 1}))
        target_q_values = self.session.run(self.target_critic([self.variables["state"], self.variables["action"]]),
                                           feed_dict={self.variables["state"]: batch.state1, self.variables["action"]: target_actions})

        # Also works
        # assert target_q_values.shape == (self.batch_size, )

        # Full the critic targets:
        # r_t + gamma * Q(s_{t + 1}, \pi(s_{t + 1}))
        discounted_reward_batch = self.gamma * target_q_values
        critic_targets = (batch.reward + discounted_reward_batch)

        feed_dict = {
            self.variables["state"]: batch.state0,
            self.variables["action"]: batch.action,
            self.critic_target: critic_targets
        }

        # Collect summaries and metrics before training the critic
        summaries = self.session.run(self.critic_summaries, feed_dict=feed_dict)

        self.metrics["critic/gradient_norm"], summaries = self.session.run([self.variables["critic/gradient_norm"], self.critic_summaries], feed_dict=feed_dict)


        # Train the critic
        for _ in range(sgd_iterations):
            # FIXME: The intermediate gradient values are not captured
            self.session.run(self.critic_train_op, feed_dict=feed_dict)

        summaries_post = self.session.run(self.critic_summaries_post, feed_dict=feed_dict)

        return (summaries, summaries_post)
