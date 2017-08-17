from __future__ import division
import os

import numpy as np
import tensorflow as tf
# Remove use of Keras backend
import keras.backend as K
from collections import namedtuple

from rl.agent import Agent
from rl.utils import clone_model, get_soft_target_model_ops
from rl.utils.numerics import gradient_inverter, huber_loss

Batch = namedtuple("Batch", ("state_0", "action", "reward", "state_1",
                             "terminal_1"))

# Whether to use Keras inference engine
USE_KERAS_INFERENCE = False


def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


class DDPGAgent(Agent):
    """
    Deep Deterministic Policy Gradient Agent as defined in https://arxiv.org/abs/1509.02971.

    :param keras.model actor: The actor network
    :param keras.model critic: The critic network
    :param gym.env env: The gym environment
    :param memory: The memory object
    :type memory: :class:`rl.memory.Memory`
    :param float gamma: Discount factor
    :param int batch_size: Size of the minibatches
    :param int nb_steps_warmup_critic: Number of warm-up steps for the critic. During these steps, there is no training
    :param int nb_steps_warmup_actor: Number of warm-up steps for the actor. During these steps, there is no training
    :param int train_interval: Train only at multiples of this number
    :param int memory_interval: Add experiences to memory only at multiples of this number
    :param delta_clip: Delta to which the rewards are clipped (via Huber loss, see https://github.com/devsisters/DQN-tensorflow/issues/16)
    :param random_process: The noise used to perform exploration
    :param custom_model_objects:
    :param float target_critic_update: Target critic update factor
    :param float target_actor_update: Target actor update factor
    :param bool invert_gradients: Use gradient inverting as defined in https://arxiv.org/abs/1511.04143
    """

    def __init__(self,
                 actor,
                 critic,
                 env,
                 memory,
                 gamma=.99,
                 batch_size=32,
                 nb_steps_warmup_critic=1000,
                 nb_steps_warmup_actor=1000,
                 train_interval=1,
                 memory_interval=1,
                 delta_clip=np.inf,
                 random_process=None,
                 custom_model_objects=None,
                 target_critic_update=1e-3,
                 # FIXME: Use 1e-4 as stated in reproducinility paper
                 target_actor_update=1e-3,
                 invert_gradients=False,
                 gradient_inverter_min=-1.,
                 gradient_inverter_max=1.,
                 actor_reset_threshold=0.3,
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

        super(DDPGAgent, self).__init__(**kwargs)

        # Get placeholders
        self.variables["state"] = env.state
        self.variables["action"] = env.action

        # Parameters.
        self.env = env
        self.nb_actions = env.action_space.dim
        self.actions_low = env.action_space.low
        self.actions_high = env.action_space.high
        self.nb_steps_warmup_actor = nb_steps_warmup_actor
        self.nb_steps_warmup_critic = nb_steps_warmup_critic
        self.random_process = random_process
        self.delta_clip = delta_clip
        self.gamma = gamma
        self.target_critic_update = process_parameterization_variable(
            target_critic_update)
        self.target_actor_update = process_parameterization_variable(
            target_actor_update)
        self.batch_size = batch_size
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.custom_model_objects = custom_model_objects
        self.invert_gradients = invert_gradients
        if invert_gradients:
            self.gradient_inverter_max = gradient_inverter_max
            self.gradient_inverter_min = gradient_inverter_min
        self.actor_reset_threshold = actor_reset_threshold

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
        # def clipped_error(y_true, y_pred):
        # return K.mean(huber_loss(y_true, y_pred, self.delta_clip), axis=-1)

        # Compile target networks. We only use them in feed-forward mode, hence we can pass any
        # optimizer and loss since we never use it anyway.
        self.target_actor = clone_model(self.actor, self.custom_model_objects)
        self.target_actor.compile(optimizer='sgd', loss='mse')
        self.target_critic = clone_model(self.critic,
                                         self.custom_model_objects)
        self.target_critic.compile(optimizer='sgd', loss='mse')

        # We also compile the actor. We never optimize the actor using Keras but instead compute
        # the policy gradient ourselves. However, we need the actor in feed-forward mode, hence
        # we also compile it with any optimizer
        self.actor.compile(optimizer='sgd', loss='mse')

        # Compile the critic for the same reason
        self.critic.compile(optimizer='sgd', loss='mse')

        # Compile the critic optimizer
        critic_optimizer = tf.train.AdamOptimizer()
        # NOT to be mistaken with the target_critic!
        self.critic_target = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        # Clip the critic gradient using the huber loss
        self.variables["critic/loss"] = K.mean(
            huber_loss(
                self.critic([self.variables["state"], self.variables["action"]]), self.critic_target,
                self.delta_clip))
        critic_gradient_vars = critic_optimizer.compute_gradients(
            self.variables["critic/loss"],
            var_list=self.critic.trainable_weights)

        # Compute the norm as a metric
        critic_gradients_norms = [
            tf.norm(grad_var[0]) for grad_var in critic_gradient_vars
        ]
        for var, norm in zip(self.critic.trainable_weights,
                             critic_gradients_norms):
            var_name = "critic/{}/gradient_norm".format(var.name)
            self.variables[var_name] = norm
        self.variables["critic/gradient_norm"] = tf.reduce_sum(
            critic_gradients_norms)

        self.critic_train_fn = critic_optimizer.apply_gradients(
            critic_gradient_vars)

        # Target critic optimizer
        if self.target_critic_update < 1.:
            # Include soft target model updates.
            self.target_critic_train_fn = get_soft_target_model_ops(
                self.target_critic.weights, self.critic.weights,
                self.target_critic_update)

        # Target actor optimizer
        if self.target_actor_update < 1.:
            # Include soft target model updates.
            self.target_actor_train_fn = get_soft_target_model_ops(
                self.target_actor.weights, self.actor.weights,
                self.target_actor_update)

        # Actor optimizer
        actor_optimizer = tf.train.AdamOptimizer()
        # Be careful to negate the gradient
        # Since the optimizer wants to minimize the value
        self.variables["actor/loss"] = -tf.reduce_mean(
            self.critic([self.variables["state"], self.actor(self.variables["state"])]))
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

        # The actual train function
        self.actor_train_fn = actor_optimizer.apply_gradients(
            actor_gradient_vars)

        # Collect summaries directly from variables
        for (var_name, variable) in self.variables.items():
            self.summary_variables[var_name] = (tf.summary.scalar(
                var_name, variable))
        # Special selections of summary variables
        # Critic
        self.critic_summaries = [
            value for (key, value) in self.summary_variables.items()
            if key.startswith("critic/")
        ]
        # Actor
        # No need to collect the actor's loss, since we already have actor/objective
        self.actor_summaries = [
            value for (key, value) in self.summary_variables.items()
            if (key.startswith("actor/") and not key == ("actor/loss"))
        ]

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
        print("Hard update of the target critic")
        self.target_critic.set_weights(self.critic.get_weights())

    def hard_update_target_actor(self):
        print("Hard update of the target actor")
        self.target_actor.set_weights(self.actor.get_weights())

    def reset_states(self):
        if self.random_process is not None:
            self.random_process.reset_states()

        if self.compiled:
            self.actor.reset_states()
            self.critic.reset_states()
            self.target_actor.reset_states()
            self.target_critic.reset_states()

    def select_action(self, state):
        # [state] is the unprocessed version of a batch
        batch_state = [state]
        # We get a batch of 1 action
        # action = self.actor.predict_on_batch(batch_state)[0]
        action = self.session.run(
            self.actor(self.variables["state"]),
            feed_dict={self.variables["state"]: batch_state,
                       K.learning_phase(): 0})[0]
        assert action.shape == (self.nb_actions, )

        # Apply noise, if a random process is set.
        if self.training and self.random_process is not None:
            noise = self.random_process.sample()
            assert noise.shape == action.shape
            action += noise

        # Clip the action value, even if the noise is making it exceed its bounds
        action = np.clip(action, self.actions_low, self.actions_high)
        return action

    def forward(self, observation):
        # Select an action.
        # state = self.memory.get_recent_state(observation)
        # action = self.select_action(state)
        action = self.select_action(observation)

        return (action)

    def backward(self,
                 offline=False,
                 fit_actor=True,
                 fit_critic=True):
        """
        Backward method of the DDPG agent

        :param bool offline: Add the new experiences to memory
        :param bool fit_actor: Activate of Deactivate training of the actor
        :param bool fit_critic: Activate of Deactivate training of the critic
        """
        # Stop here if not training
        if not self.training:
            return

        # Store most recent experience in memory.
        # Nothing to store in offline mode
        if not offline:
            if self.step % self.memory_interval == 0:
                self.memory.append(self.observation, self.action, self.reward,
                                   self.observation_1, self.done)

        # Train the networks
        if self.step % self.train_interval == 0:

            # If warm up is over:
            # Update critic
            fit_critic = (fit_critic and
                          self.step > self.nb_steps_warmup_critic)
            # Update actor
            fit_actor = (fit_actor and self.step > self.nb_steps_warmup_actor)

            # Hard update the target nets if necessary
            if self.target_actor_update >= 1:
                hard_update_target_actor = self.step % self.target_actor_update == 0
            else:
                hard_update_target_actor = False
            if self.target_critic_update >= 1:
                hard_update_target_critic = self.step % self.target_critic_update == 0
            else:
                hard_update_target_critic = False

            # Whether to reset the actor
            if self.done and (self.episode % 5 == 0):
                can_reset_actor = True
            else:
                can_reset_actor = False

            if (fit_actor or fit_critic):
                self.fit_controllers(
                    fit_critic=fit_critic,
                    fit_actor=fit_actor,
                    can_reset_actor=can_reset_actor,
                    hard_update_target_critic=hard_update_target_critic,
                    hard_update_target_actor=hard_update_target_actor)

    def fit_controllers(self,
                        fit_critic=True,
                        fit_actor=True,
                        can_reset_actor=False,
                        hard_update_target_critic=False,
                        hard_update_target_actor=False):
        """
        Fit the actor and critic networks

        :param bool fit_critic: Whether to fit the critic
        :param bool fit_actor: Whether to fit the actor
        :param bool can_reset_actor:

        """

        if not (fit_actor or fit_critic):
            return
        else:
            batch = self.get_batch()

            summaries = []

            # Train networks
            if fit_critic:
                summaries_critic = self.fit_critic(batch)
                summaries += summaries_critic

            if fit_actor:
                summaries_actor = self.fit_actor(
                    batch, can_reset_actor=can_reset_actor)
                summaries += summaries_actor

            # Update target networks
            if self.target_actor_update >= 1:
                if hard_update_target_actor:
                    self.hard_update_target_actor()
            else:
                self.session.run(self.target_actor_train_fn)
            if self.target_critic_update >= 1:
                if hard_update_target_critic:
                    self.hard_update_target_critic()
            else:
                self.session.run(self.target_critic_train_fn)

            self.step_summaries += summaries

    def fit_critic(self, batch, sgd_iterations=1):
        """Fit the critic network"""
        # Get the target action
        # \pi(s_t)
        if USE_KERAS_INFERENCE:
            target_actions = self.target_actor.predict_on_batch(batch.state_1)
        else:
            target_actions = self.session.run(
                self.target_actor(self.variables["state"]),
                feed_dict={self.variables["state"]: batch.state_1,
                           K.learning_phase(): 0})
        assert target_actions.shape == (self.batch_size, self.nb_actions)

        # Get the target Q value
        # Q(s_t, \pi(s_t))
        if USE_KERAS_INFERENCE:
            target_q_values = self.target_critic.predict_on_batch(
                [batch.state_1, target_actions]).flatten()
        else:
            target_q_values = self.session.run(
                self.target_critic([self.variables["state"], self.variables["action"]]),
                feed_dict={
                    self.variables["state"]: batch.state_1,
                    self.variables["action"]: target_actions
                }).flatten()

        # Also works
        assert target_q_values.shape == (self.batch_size, )

        # Compute the critic targets:
        # r_t + gamma * Q(s_t, \pi(s_t))
        # but only for the affected output units (as given by action_batch).
        discounted_reward_batch = self.gamma * target_q_values
        discounted_reward_batch *= batch.terminal_1
        critic_targets = (batch.reward + discounted_reward_batch).reshape(
            self.batch_size, 1)

        feed_dict = {
            self.variables["state"]: batch.state_0,
            self.variables["action"]: batch.action,
            self.critic_target: critic_targets
        }

        # Collect summaries and metrics before training the critic
        summaries = self.session.run(
            self.critic_summaries, feed_dict=feed_dict)

        # Train the critic
        for _ in range(sgd_iterations):
            # FIXME: The intermediate gradient values are not captured
            self.session.run(self.critic_train_fn, feed_dict=feed_dict)

        return (summaries)

    def fit_actor(self, batch, sgd_iterations=1, can_reset_actor=False):
        """Fit the actor network"""

        feed_dict = {self.variables["state"]: batch.state_0, K.learning_phase(): 1}

        # Collect metrics before training the actor
        self.metrics["actor/gradient_norm"], summaries = self.session.run(
            [self.variables["actor/gradient_norm"], self.actor_summaries],
            feed_dict=feed_dict)

        # Train the actor
        for _ in range(sgd_iterations):
            # FIXME: The intermediate gradient values are not captured
            self.session.run(self.actor_train_fn, feed_dict=feed_dict)

        if can_reset_actor:
            # Reset the actor if the gradient is flat
            if self.metrics["actor/gradient_norm"] <= self.actor_reset_threshold:
                # TODO: Use a gradient on a rolling window: multiple steps (and even multiple episodes)
                self.restore_checkpoint(actor=True, critic=False)

        return (summaries)

    def get_batch(self):
        """
        Get and process a batch
        Split each batch of experiences into batches of state_0, action etc...
        """
        # TODO: Remove this function
        # Store directly the different batches into memory
        experiences = self.memory.sample(self.batch_size)
        assert len(experiences) == self.batch_size

        # Start by extracting the necessary parameters (we use a vectorized implementation).
        state0_batch = []
        reward_batch = []
        action_batch = []
        terminal1_batch = []
        state1_batch = []
        for e in experiences:
            state0_batch.append(e.state0)
            state1_batch.append(e.state1)
            reward_batch.append(e.reward)
            action_batch.append(e.action)
            terminal1_batch.append(0. if e.terminal1 else 1.)

        # Prepare and validate parameters.
        state0_batch = np.array(state0_batch)
        state1_batch = np.array(state1_batch)
        terminal1_batch = np.array(terminal1_batch)
        reward_batch = np.array(reward_batch)
        action_batch = np.array(action_batch)

        batch = Batch(
            state_0=state0_batch,
            action=action_batch,
            reward=reward_batch,
            terminal_1=terminal1_batch,
            state_1=state1_batch)
        return (batch)

    def checkpoint(self):
        """Save the weights"""
        self.checkpoints.append((self.actor.get_weights(),
                                 self.critic.get_weights()))

    def restore_checkpoint(self, actor=True, critic=True, checkpoint_id=0):
        """Restore from checkpoint"""
        weights_actor, weights_critic = self.checkpoints[checkpoint_id]
        if actor:
            print("\033[31mRestoring actor\033[0m")
            self.actor.set_weights(weights_actor)
            self.target_actor.set_weights(weights_actor)
        if critic:
            print("\033[31mRestoring critic\033[0m")
            self.critic.set_weights(weights_critic)
            self.target_critic.set_weights(weights_critic)


def process_parameterization_variable(param):
    # Soft vs hard target model updates.
    if param < 0:
        raise ValueError('`target_model_update` must be >= 0, currently at {}'.
                         format(param))
    elif param >= 1:
        # Hard update every `target_model_update` steps.
        param = int(param)
    else:
        # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
        param = float(param)
    return (param)
