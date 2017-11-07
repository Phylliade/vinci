from __future__ import division
import os

import numpy as np
import tensorflow as tf
# Remove use of Keras backend
import keras.backend as K

from rl.utils.model import clone_model, get_soft_target_model_ops
from rl.utils.numerics import gradient_inverter, huber_loss
from rl.memory import Experience
from rl.agents.rlagent import RLAgent
from rl.utils.printer import print_warning

# Whether to use Keras inference engine
USE_KERAS_INFERENCE = False


class DDPGAgent(RLAgent):
    """
    Deep Deterministic Policy Gradient Agent as defined in https://arxiv.org/abs/1509.02971.

    :param keras.model actor: The actor network
    :param keras.model critic: The critic network
    :param gym.env env: The gym environment
    :param memory: The memory object
    :type memory: :class:`rl.memory.Memory`
    :param float gamma: Discount factor
    :param int batch_size: Size of the minibatches
    :param int train_interval: Train only at multiples of this number
    :param int memory_interval: Add experiences to memory only at multiples of this number
    :param critic_gradient_clip: Delta to which the rewards are clipped (via Huber loss, see https://github.com/devsisters/DQN-tensorflow/issues/16)
    :param random_process: The noise used to perform exploration
    :param custom_model_objects:
    :param float target_critic_update: Target critic update factor
    :param float target_actor_update: Target actor update factor
    :param bool invert_gradients: Use gradient inverting as defined in https://arxiv.org/abs/1511.04143
    """

    def __init__(self,
                 actor,
                 critic,
                 memory,
                 gamma=.99,
                 batch_size=32,
                 train_interval=1,
                 memory_interval=1,
                 critic_gradient_clip=100,
                 random_process=None,
                 custom_model_objects=None,
                 warmup_actor_steps=200,
                 warmup_critic_steps=200,
                 invert_gradients=False,
                 gradient_inverter_min=-1.,
                 gradient_inverter_max=1.,
                 actor_reset_threshold=0.3,
                 reset_controlers=False,
                 actor_learning_rate=1e-3,
                 critic_learning_rate=1e-4,
                 target_critic_update=0.01,
                 target_actor_update=0.01,
                 critic_regularization=0.01,
                 **kwargs):

        if custom_model_objects is None:
            custom_model_objects = {}
        if hasattr(actor.output, '__len__') and len(actor.output) > 1:
            raise ValueError(
                'Actor "{}" has more than one output. DDPG expects an actor that has a single output.'.
                format(actor))
        if hasattr(critic.output, '__len__') and len(critic.output) > 1:
            raise ValueError(
                'Critic "{}" has more than one output. DDPG expects a critic that has a single output.'.
                format(critic))
        if not hasattr(critic.input, '__len__') or len(critic.input) < 2:
            raise ValueError(
                'Critic "{}" does not have enough inputs. The critic must have at exactly two inputs, one for the action and one for the observation.'.
                format(critic))

        super(DDPGAgent, self).__init__(name="ddpg", **kwargs)

        print("delta-clip :", critic_gradient_clip)

        # Get placeholders
        self.variables["state"] = self.env.state
        self.variables["action"] = self.env.action

        # Parameters.
        self.nb_actions = self.env.action_space.dim
        self.actions_low = self.env.action_space.low
        self.actions_high = self.env.action_space.high
        self.random_process = random_process
        self.critic_gradient_clip = critic_gradient_clip
        self.gamma = gamma
        self.warmup_actor_steps = warmup_actor_steps
        self.warmup_critic_steps = warmup_critic_steps
        self.critic_learning_rate = critic_learning_rate
        self.actor_learning_rate = actor_learning_rate
        self.critic_regularization = critic_regularization
        (self.target_critic_update, self.target_critic_hard_updates
         ) = process_hard_update_variable(target_critic_update)
        (self.target_actor_update, self.target_actor_hard_updates
         ) = process_hard_update_variable(target_actor_update)
        self.batch_size = batch_size
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.custom_model_objects = custom_model_objects
        self.invert_gradients = invert_gradients
        if invert_gradients:
            self.gradient_inverter_max = gradient_inverter_max
            self.gradient_inverter_min = gradient_inverter_min
        self.actor_reset_threshold = actor_reset_threshold
        self.reset_controlers = reset_controlers

        # Related objects.
        self.actor = actor
        self.critic = critic
        self.memory = memory

        # State.
        self.compiled = False
        self.reset_states()

    @property
    def uses_learning_phase(self):
        return self.actor.uses_learning_phase or self.critic.uses_learning_phase

    def load_memory(self, memory):
        """Loads the given memory as the replay buffer"""
        del (self.memory)
        self.memory = memory

    def compile(self):
        # Compile target networks. We only use them in feed-forward mode, hence we can pass any
        # optimizer and loss since we never use it anyway.
        self.target_actor = clone_model(self.actor, self.custom_model_objects)
        self.target_actor.compile(optimizer='sgd', loss='mse')
        self.target_critic = clone_model(self.critic,
                                         self.custom_model_objects)
        self.target_critic.compile(optimizer='sgd', loss='mse')

        self.compile_actor()
        self.compile_critic()

        # Collect summaries directly from variables
        for (var_name, variable) in self.variables.items():
            self.summary_variables[var_name] = (tf.summary.scalar(
                var_name, variable))
        # Special selections of summary variables
        # Critic
        self.critic_summaries = [
            value for (key, value) in self.summary_variables.items()
            if (key.startswith("critic/") or key.startswith("target_critic/"))
        ]
        # Critic post (run after training)
        self.critic_summaries_post = [
            value for (key, value) in self.summary_variables.items()
            if (key.startswith("critic_post/")
                or key.startswith("target_critic_post/"))
        ]
        # Actor
        # No need to collect the actor's loss, since we already have actor/objective
        self.actor_summaries = [
            value for (key, value) in self.summary_variables.items()
            if (key.startswith("actor/") and not key == ("actor/loss")
                or key.startswith("target_actor/"))
        ]
        # Actor post
        self.actor_summaries_post = [
            value for (key, value) in self.summary_variables.items()
            if (key.startswith("actor_post/")
                or key.startswith("target_actor_post/"))
        ]

        # Initialize the remaining variables
        # FIXME: Use directly Keras backend
        # This is a kind of a hack
        # Taken from the "initialize_variables" of the Keras Tensorflow backend
        # https://github.com/fchollet/keras/blob/master/keras/backend/tensorflow_backend.py#L330
        # It permits to only initialize variables that are not already initialized
        # Without that, the networks and target networks get initialized again, to different values (stochastic initialization)
        # This is a problem when a network and it's target network do not begin with the same parameter values...
        variables = tf.global_variables()
        uninitialized_variables = []
        for v in variables:
            if not hasattr(v,
                           '_keras_initialized') or not v._keras_initialized:
                uninitialized_variables.append(v)
                v._keras_initialized = True
        self.session.run(tf.variables_initializer(uninitialized_variables))
        # self.session.run(tf.global_variables_initializer())

        # Save the initial values of the networks
        self.checkpoint()

        self.compiled = True

    def compile_actor(self):
        # We also compile the actor. We never optimize the actor using Keras but instead compute
        # the policy gradient ourselves. However, we need the actor in feed-forward mode, hence
        # we also compile it with any optimizer
        self.actor.compile(optimizer='sgd', loss='mse')

        # Target actor optimizer
        if not self.target_actor_hard_updates:
            # Include soft target model updates.
            self.target_actor_train_op = get_soft_target_model_ops(
                self.target_actor.weights, self.actor.weights,
                self.target_actor_update)

        # Actor optimizer
        actor_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.actor_learning_rate)
        # Be careful to negate the gradient
        # Since the optimizer wants to minimize the value
        self.variables["actor/loss"] = -tf.reduce_mean(
            self.critic(
                [self.variables["state"],
                 self.actor(self.variables["state"])]))
        self.variables["actor/objective"] = -self.variables["actor/loss"]

        actor_gradient_vars = actor_optimizer.compute_gradients(
            self.variables["actor/loss"],
            var_list=self.actor.trainable_weights)
        # Gradient inverting
        # as described in https://arxiv.org/abs/1511.04143
        if self.invert_gradients:
            actor_gradient_vars = [(gradient_inverter(
                x[0], self.gradient_inverter_min, self.gradient_inverter_max),
                                    x[1]) for x in actor_gradient_vars]

        # Compute the norm of each weights's gradient
        actor_gradients_norms = [
            tf.norm(grad_var[0]) for grad_var in actor_gradient_vars
        ]
        for var, norm in zip(self.actor.trainable_weights,
                             actor_gradients_norms):
            var_name = "actor/{}/gradient_norm".format(var.name)
            self.variables[var_name] = (norm)
        # As long as the sum
        self.variables["actor/gradient_norm"] = tf.reduce_sum(
            actor_gradients_norms)

        # The actual train op
        self.actor_train_op = actor_optimizer.apply_gradients(
            actor_gradient_vars)

        # Additional actor metrics
        actor_norms = [
            tf.norm(weight) for weight in self.actor.trainable_weights
        ]
        for var, norm in zip(self.actor.trainable_weights, actor_norms):
            var_name = "actor/{}/norm".format(var.name)
            self.variables[var_name] = norm
        self.variables["actor/norm"] = tf.reduce_sum(actor_norms)

        # Additional target actor metrics
        target_actor_norms = [
            tf.norm(weight) for weight in self.target_actor.trainable_weights
        ]
        for var, norm in zip(self.target_critic.trainable_weights,
                             target_actor_norms):
            var_name = "target_actor/{}/norm".format(var.name)
            self.variables[var_name] = norm
        self.variables["target_actor/norm"] = tf.reduce_sum(target_actor_norms)

    def compile_critic(self):
        # Compile the critic for the same reason
        self.critic.compile(optimizer='sgd', loss='mse')

        # Compile the critic optimizer
        critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_learning_rate)
        # NOT to be mistaken with the target_critic!
        self.critic_target = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        # Clip the critic gradient using the huber loss
        self.variables["critic/loss"] = K.mean(huber_loss(self.critic([self.variables["state"], self.variables["action"]]),self.critic_target, self.critic_gradient_clip))

        # L2 regularization on the critic loss
        critic_norms = [tf.norm(weight) for weight in self.critic.trainable_weights]
        critic_norms_l2 = [tf.nn.l2_loss(weight) for weight in self.critic.trainable_weights]
        self.variables["critic/norm"] = tf.reduce_sum(critic_norms)
        self.variables["critic/l2_norm"] = tf.reduce_sum(critic_norms_l2)
        if self.critic_regularization != 0:
            self.variables["critic/loss"] += self.critic_regularization * self.variables["critic/l2_norm"]

        #  Compute gradients
        critic_gradient_vars = critic_optimizer.compute_gradients(
            self.variables["critic/loss"],
            var_list=self.critic.trainable_weights)

        # Compute the norm as a metric
        critic_gradients_norms = [tf.norm(grad_var[0]) for grad_var in critic_gradient_vars]
        for var, norm in zip(self.critic.trainable_weights, critic_gradients_norms):
            var_name = "critic/{}/gradient_norm".format(var.name)
            self.variables[var_name] = norm
        self.variables["critic/gradient_norm"] = tf.reduce_sum(critic_gradients_norms)

        # Additional global critic metrics
        self.variables["critic/distance_to_target_pre"] = tf.reduce_mean(self.critic([self.variables["state"], self.variables["action"]]) - self.target_critic([self.variables["state"], self.variables["action"]]))
        self.variables["critic/mean_val_pre"] = tf.reduce_mean(self.critic([self.variables["state"], self.variables["action"]]))
        
        self.critic_train_op = critic_optimizer.apply_gradients(critic_gradient_vars)

        # Additional global critic metrics
        self.variables["critic/distance_to_target_post"] = tf.reduce_mean(self.critic([self.variables["state"], self.variables["action"]]) - self.target_critic([self.variables["state"], self.variables["action"]]))
        self.variables["critic/mean_val_post"] = tf.reduce_mean(self.critic([self.variables["state"], self.variables["action"]]))

        # Additional critic metrics
        for var, norm in zip(self.critic.trainable_weights, critic_norms):
            var_name = "critic/{}/norm".format(var.name)
            self.variables[var_name] = norm

        # Additional target critic metrics
        target_critic_norms = [tf.norm(weight) for weight in self.target_critic.trainable_weights]
        for var, norm in zip(self.target_critic.trainable_weights, target_critic_norms):
            var_name = "target_critic/{}/norm".format(var.name)
            self.variables[var_name] = norm
        self.variables["target_critic/norm"] = tf.reduce_sum(target_critic_norms)

        # Target critic optimizer
        if not self.target_critic_hard_updates:
            # Include soft target model updates.
            self.target_critic_train_op = get_soft_target_model_ops(self.target_critic.weights, self.critic.weights, self.target_critic_update)

    def load_weights(self, filepath):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.load_weights(actor_filepath)
        self.critic.load_weights(critic_filepath)
        self.hard_update_target_models()

    def save_weights(self, filepath, overwrite=False):
        print("Saving weights")
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.save_weights(actor_filepath, overwrite=overwrite)
        self.critic.save_weights(critic_filepath, overwrite=overwrite)

    def save(self, name="DDPG"):
        """Save the model as an HDF5 file"""
        self.actor.save(name + "_actor.h5")
        self.critic.save(name + "_critic.h5")

    def hard_update_target_critic(self):
        #print("Hard update of the target critic")
        self.target_critic.set_weights(self.critic.get_weights())

    def hard_update_target_actor(self):
        #print("Hard update of the target actor")
        self.target_actor.set_weights(self.actor.get_weights())

    def reset_states(self):
        if self.random_process is not None:
            self.random_process.reset_states()

        if self.compiled:
            self.actor.reset_states()
            self.critic.reset_states()
            self.target_actor.reset_states()
            self.target_critic.reset_states()

    def forward(self, observation):
        # Select an action.
        # [state] is the unprocessed version of a batch
        batch_state = [observation]
        # We get a batch of 1 action
        # action = self.actor.predict_on_batch(batch_state)[0]
        action = self.session.run(
            self.actor(self.variables["state"]),
            feed_dict={
                self.variables["state"]: batch_state,
                K.learning_phase(): 0
            })[0]
        assert action.shape == (self.nb_actions, )

        # Apply noise, if a random process is set.
        if self.exploration and self.random_process is not None:
            noise = self.random_process.sample()
            assert noise.shape == action.shape
            action += noise

        # Clip the action value, even if the noise is making it exceed its bounds
        action = np.clip(action, self.actions_low, self.actions_high)
        return action

        return (action)

    def backward(self):
        """
        Backward method of the DDPG agent
        """
        # Stop here if not training
        if not self.training:
            return

        # Store most recent experience in memory.
        if self.training_step % self.memory_interval == 0:
            self.memory.append(
                Experience(self.observation, self.action,
                           self.reward, self.observation_1, self.done))

        # Train the networks
        if self.training_step % self.train_interval == 0:
            # If warm up is over:
            # Update critic
            self.warmingup_critic = (self.training_step <= self.warmup_critic_steps)
            train_critic = (not self.warmingup_critic)
            # Update actor
            self.warmingup_actor = self.training_step <= self.warmup_actor_steps
            train_actor = (not self.warmingup_actor)

            self._backward(train_actor=train_actor, train_critic=train_critic)

    def backward_offline(self, train_actor=True, train_critic=True):
        """
        Offline Backward method of the DDPG agent

        :param bool offline: Add the new experiences to memory
        :param bool train_actor: Activate of Deactivate training of the actor
        :param bool train_critic: Activate of Deactivate training of the critic
        """
        # Stop here if not training
        if not self.training:
            return

        self._backward(train_actor=train_actor, train_critic=train_critic)

    def _backward(self, train_actor=True, train_critic=True):
        """
        Offline Backward method of the DDPG agent

        :param bool offline: Add the new experiences to memory
        :param bool train_actor: Activate of Deactivate training of the actor
        :param bool train_critic: Activate of Deactivate training of the critic
        """
        # Stop here if not training
        if not self.training:
            return

        # Train the networks
        if self.training_step % self.train_interval == 0:

            # Hard update the target nets if necessary
            if self.target_actor_hard_updates:
                hard_update_target_actor = self.training_step % self.target_actor_update == 0
            else:
                hard_update_target_actor = False

            if self.target_critic_hard_updates:
                hard_update_target_critic = self.training_step % self.target_critic_update == 0
            else:
                hard_update_target_critic = False

            # Whether to reset the actor
            if self.done and (self.episode % 5 == 0) and self.reset_controlers:
                can_reset_actor = True
            else:
                can_reset_actor = False

            if (train_actor or train_critic):
                self.train_controllers(
                    train_critic=train_critic,
                    train_actor=train_actor,
                    can_reset_actor=can_reset_actor,
                    hard_update_target_critic=hard_update_target_critic,
                    hard_update_target_actor=hard_update_target_actor)

    def train_controllers(self,
                          train_critic=True,
                          train_actor=True,
                          can_reset_actor=False,
                          hard_update_target_critic=False,
                          hard_update_target_actor=False):
        """
        Fit the actor and critic networks

        :param bool train_critic: Whether to fit the critic
        :param bool train_actor: Whether to fit the actor
        :param bool can_reset_actor:

        """

        if not (train_actor or train_critic):
            return
        else:
            batch = self.memory.sample(self.batch_size)

            summaries = []
            summaries_post = []

            # Train networks
            if train_critic:
                summaries_critic, summaries_post_critic = self.train_critic(
                    batch)
                summaries += summaries_critic
                summaries_post += summaries_post_critic

            if train_actor:
                summaries_actor, summaries_post_actor = self.train_actor(
                    batch, can_reset_actor=can_reset_actor)
                summaries += summaries_actor
                summaries_post += summaries_post_actor

            # Update target networks
            if hard_update_target_actor:
                self.hard_update_target_actor()
            else:
                self.session.run(self.target_actor_train_op)
            if hard_update_target_critic:
                self.hard_update_target_critic()
            else:
                self.session.run(self.target_critic_train_op)

            self.step_summaries += summaries
            self.step_summaries_post += summaries_post

    def train_critic(self, batch, sgd_iterations=1):
        """Fit the critic network"""
        # Get the target action
        # \pi(s_{t + 1})
        if USE_KERAS_INFERENCE:
            target_actions = self.target_actor.predict_on_batch(batch.state1)
        else:
            target_actions = self.session.run(
                self.target_actor(self.variables["state"]),
                feed_dict={
                    self.variables["state"]: batch.state1,
                    K.learning_phase(): 0
                })
        assert target_actions.shape == (self.batch_size, self.nb_actions)

        # Get the target Q value of the next state
        # Q(s_{t + 1}, \pi(s_{t + 1}))
        if USE_KERAS_INFERENCE:
            target_q_values = self.target_critic.predict_on_batch(
                [batch.state1, target_actions])
        else:
            target_q_values = self.session.run(
                self.target_critic(
                    [self.variables["state"], self.variables["action"]]),
                feed_dict={
                    self.variables["state"]: batch.state1,
                    self.variables["action"]: target_actions
                })

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
        self.metrics["critic/gradient_norm"], summaries = self.session.run(
            [self.variables["critic/gradient_norm"], self.critic_summaries], feed_dict=feed_dict)

        # Train the critic
        for _ in range(sgd_iterations):
            # FIXME: The intermediate gradient values are not captured
            self.session.run(self.critic_train_op, feed_dict=feed_dict)

            # Collect summaries and metrics after training the critic
        summaries_post = self.session.run(
            self.critic_summaries_post, feed_dict=feed_dict)

        return (summaries, summaries_post)

    def train_actor(self, batch, sgd_iterations=1, can_reset_actor=False):
        """Fit the actor network"""

        feed_dict = {
            self.variables["state"]: batch.state0,
            K.learning_phase(): 1
        }

        # Collect metrics before training the actor
        self.metrics["actor/gradient_norm"], summaries = self.session.run(
            [self.variables["actor/gradient_norm"], self.actor_summaries],
            feed_dict=feed_dict)

        # Train the actor
        for _ in range(sgd_iterations):
            # FIXME: The intermediate gradient values are not captured
            self.session.run(self.actor_train_op, feed_dict=feed_dict)

        # Collect metrics before training the actor
        summaries_post = self.session.run(
            self.actor_summaries_post, feed_dict=feed_dict)

        if can_reset_actor:
            # Reset the actor if the gradient is flat
            if self.metrics["actor/gradient_norm"] <= self.actor_reset_threshold:
                # TODO: Use a gradient on a rolling window: multiple steps (and even multiple episodes)
                self.restore_checkpoint(actor=True, critic=False)

        return (summaries, summaries_post)

    def checkpoint(self):
        """Save the weights"""
        self.checkpoints.append((self.actor.get_weights(),
                                 self.critic.get_weights()))

    def restore_checkpoint(self, actor=True, critic=True, checkpoint_id=0):
        """Restore from checkpoint"""
        weights_actor, weights_critic = self.checkpoints[checkpoint_id]
        if actor:
            print_warning("Restoring actor and target actor")
            self.actor.set_weights(weights_actor)
            self.target_actor.set_weights(weights_actor)
        if critic:
            print_warning("Restoring critic")
            self.critic.set_weights(weights_critic)
            self.target_critic.set_weights(weights_critic)


def process_hard_update_variable(param):
    # Soft vs hard target model updates.
    if param < 0:
        raise ValueError(
            '`target_model_update` must be >= 0, currently at {}'.format(
                param))
    elif param >= 1:
        # Hard update every `target_model_update` steps.
        param = int(param)
        hard_updates = True
    else:
        # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
        param = float(param)
        hard_updates = False
    return (param, hard_updates)
