from __future__ import division
import os
import warnings

import numpy as np
import keras.backend as K
import keras.optimizers as optimizers
from collections import namedtuple

from rl.core import Agent
from rl.util import huber_loss, clone_model, get_soft_target_model_updates, clone_optimizer, AdditionalUpdatesOptimizer

Batches = namedtuple("Batches", ("state_0", "action", "reward", "state_1",
                                 "terminal_1"))


def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


# Deep DPG as described by Lillicrap et al. (2015)
# http://arxiv.org/pdf/1509.02971v2.pdf
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.646.4324&rep=rep1&type=pdf
class DDPGAgent(Agent):
    """Write me
    """

    def __init__(
        self,
        nb_actions,
        actions_low,
        actions_high,
        actor,
        critic,
        critic_action_input,
        memory,
        gamma=.99,
        batch_size=32,
        nb_steps_warmup_critic=1000,
        nb_steps_warmup_actor=1000,
        train_interval=1,
        memory_interval=1,
        delta_range=None,
        delta_clip=np.inf,
        random_process=None,
        custom_model_objects={},
        target_critic_update=.001,
        target_actor_update=0.001,
        **kwargs
    ):
        if hasattr(actor.output, '__len__') and len(actor.output) > 1:
            raise ValueError(
                'Actor "{}" has more than one output. DDPG expects an actor that has a single output.'.
                format(actor))
        if hasattr(critic.output, '__len__') and len(critic.output) > 1:
            raise ValueError(
                'Critic "{}" has more than one output. DDPG expects a critic that has a single output.'.
                format(critic))
        if critic_action_input not in critic.input:
            raise ValueError(
                'Critic "{}" does not have designated action input "{}".'.
                format(critic, critic_action_input))
        if not hasattr(critic.input, '__len__') or len(critic.input) < 2:
            raise ValueError(
                'Critic "{}" does not have enough inputs. The critic must have at exactly two inputs, one for the action and one for the observation.'.
                format(critic))

        super(DDPGAgent, self).__init__(**kwargs)

        if delta_range is not None:
            warnings.warn(
                '`delta_range` is deprecated. Please use `delta_clip` instead, which takes a single scalar. For now we\'re falling back to `delta_range[1] = {}`'.
                format(delta_range[1]))
            delta_clip = delta_range[1]

        # Parameters.
        self.nb_actions = nb_actions
        self.actions_low = actions_low
        self.actions_high = actions_high
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

        # Related objects.
        self.actor = actor
        self.critic = critic
        self.critic_action_input = critic_action_input
        self.critic_action_input_idx = self.critic.input.index(
            critic_action_input)
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

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]

        if type(optimizer) in (list, tuple):
            if len(optimizer) != 2:
                raise ValueError(
                    'More than two optimizers provided. Please only provide a maximum of two optimizers, the first one for the actor and the second one for the critic.'
                )
            actor_optimizer, critic_optimizer = optimizer
        else:
            actor_optimizer = optimizer
            critic_optimizer = clone_optimizer(optimizer)
        if type(actor_optimizer) is str:
            actor_optimizer = optimizers.get(actor_optimizer)
        if type(critic_optimizer) is str:
            critic_optimizer = optimizers.get(critic_optimizer)
        assert actor_optimizer != critic_optimizer

        if len(metrics) == 2 and hasattr(metrics[0], '__len__') and hasattr(
                metrics[1], '__len__'):
            actor_metrics, critic_metrics = metrics
        else:
            actor_metrics = critic_metrics = metrics

        def clipped_error(y_true, y_pred):
            return K.mean(huber_loss(y_true, y_pred, self.delta_clip), axis=-1)

        # Compile target networks. We only use them in feed-forward mode, hence we can pass any
        # optimizer and loss since we never use it anyway.
        self.target_actor = clone_model(self.actor, self.custom_model_objects)
        self.target_actor.compile(optimizer='sgd', loss='mse')
        self.target_critic = clone_model(self.critic,
                                         self.custom_model_objects)
        self.target_critic.compile(optimizer='sgd', loss='mse')

        # We also compile the actor. We never optimize the actor using Keras but instead compute
        # the policy gradient ourselves. However, we need the actor in feed-forward mode, hence
        # we also compile it with any optimizer and
        self.actor.compile(optimizer='sgd', loss='mse')

        # Compile the critic.
        if self.target_critic_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            critic_updates = get_soft_target_model_updates(
                self.target_critic, self.critic, self.target_critic_update)
            critic_optimizer = AdditionalUpdatesOptimizer(
                critic_optimizer, critic_updates)
        self.critic.compile(
            optimizer=critic_optimizer,
            loss=clipped_error,
            metrics=critic_metrics)

        # Combine actor and critic so that we can get the policy gradient.
        combined_inputs = []
        critic_inputs = []
        for i in self.critic.input:
            if i == self.critic_action_input:
                combined_inputs.append(self.actor.output)
            else:
                combined_inputs.append(i)
                critic_inputs.append(i)
        combined_output = self.critic(combined_inputs)
        if K.backend() == 'tensorflow':
            grads = K.gradients(combined_output, self.actor.trainable_weights)
            grads = [g / float(self.batch_size)
                     for g in grads]  # since TF sums over the batch
        elif K.backend() == 'theano':
            import theano.tensor as T
            grads = T.jacobian(combined_output.flatten(),
                               self.actor.trainable_weights)
            grads = [K.mean(g, axis=0) for g in grads]
        else:
            raise RuntimeError(
                'Unknown Keras backend "{}".'.format(K.backend()))

        # We now have the gradients (`grads`) of the combined model wrt to the actor's weights and
        # the output (`output`). Compute the necessary updates using a clone of the actor's optimizer.
        clipnorm = getattr(actor_optimizer, 'clipnorm', 0.)
        clipvalue = getattr(actor_optimizer, 'clipvalue', 0.)

        def get_gradients(loss, params):
            # We want to follow the gradient, but the optimizer goes in the opposite direction to
            # minimize loss. Hence the double inversion.
            assert len(grads) == len(params)
            modified_grads = [-g for g in grads]
            if clipnorm > 0.:
                norm = K.sqrt(
                    sum([K.sum(K.square(g)) for g in modified_grads]))
                modified_grads = [
                    optimizers.clip_norm(g, clipnorm, norm)
                    for g in modified_grads
                ]
            if clipvalue > 0.:
                modified_grads = [
                    K.clip(g, -clipvalue, clipvalue) for g in modified_grads
                ]
            return modified_grads

        actor_optimizer.get_gradients = get_gradients
        updates = actor_optimizer.get_updates(self.actor.trainable_weights,
                                              self.actor.constraints, None)
        if self.target_actor_update < 1.:
            # Include soft target model updates.
            updates += get_soft_target_model_updates(
                self.target_actor, self.actor, self.target_actor_update)
        updates += self.actor.updates  # include other updates of the actor, e.g. for BN

        # Finally, combine it all into a callable function.
        inputs = self.actor.inputs[:] + critic_inputs
        if self.uses_learning_phase:
            inputs += [K.learning_phase()]

        # Function to train the actor
        self.actor_train_fn = K.function(
            inputs, [self.actor.output], updates=updates)
        self.actor_optimizer = actor_optimizer

        self.compiled = True

    def load_weights(self, filepath):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.load_weights(actor_filepath)
        self.critic.load_weights(critic_filepath)
        self.update_target_models_hard()

    def save_weights(self, filepath, overwrite=False):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.save_weights(actor_filepath, overwrite=overwrite)
        self.critic.save_weights(critic_filepath, overwrite=overwrite)

    def update_target_critic_hard(self):
        self.target_critic.set_weights(self.critic.get_weights())

    def update_target_actor_hard(self):
        self.target_actor.set_weights(self.actor.get_weights())

    # TODO: implement pickle

    def reset_states(self):
        if self.random_process is not None:
            self.random_process.reset_states()

        if self.compiled:
            self.actor.reset_states()
            self.critic.reset_states()
            self.target_actor.reset_states()
            self.target_critic.reset_states()

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def select_action(self, state):
        batch = self.process_state_batch([state])
        action = self.actor.predict_on_batch(batch).flatten()
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
        action = self.select_action([observation])

        action = self.processor.process_action(action)

        return action

    @property
    def layers(self):
        return self.actor.layers[:] + self.critic.layers[:]

    @property
    def metrics_names(self):
        names = self.critic.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

    def backward(self,
                 observations,
                 action,
                 reward,
                 terminal=False,
                 fit_actor=True,
                 fit_critic=True):
        metrics = [np.nan for _ in self.metrics_names]

        # Stop here if not training
        if not self.training:
            return metrics

        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(observations[0], action, reward,
                               observations[1], terminal)

        # Train the network on a single stochastic batch.
        can_train_either = self.step > self.nb_steps_warmup_critic or self.step > self.nb_steps_warmup_actor

        if can_train_either and self.step % self.train_interval == 0:
            metrics = self.fit_nets(fit_critic=fit_critic, fit_actor=fit_actor)

        return metrics

    def fit_nets(self, fit_critic=True, fit_actor=True):
        batches = self.process_batches()

        # Update critic, if warm up is over.
        if fit_critic and self.step > self.nb_steps_warmup_critic:
            metrics = self.fit_critic(batches)

        # Update actor, if warm up is over.
        if fit_actor and self.step > self.nb_steps_warmup_actor:
            self.fit_actor(batches)

        # Update target networks
        if self.target_critic_update >= 1 and self.step % self.target_critic_update == 0:
            self.update_target_critic_hard()
        # Update target networks
        if self.target_actor_update >= 1 and self.step % self.target_actor_update == 0:
            self.update_target_actor_hard()

        return metrics

    def fit_critic(self, batches):
        target_actions = self.target_actor.predict_on_batch(batches.state_1)
        assert target_actions.shape == (self.batch_size, self.nb_actions)
        if len(self.critic.inputs) >= 3:
            state1_batch_with_action = batches
        else:
            state1_batch_with_action = [batches.state_1]
        state1_batch_with_action.insert(self.critic_action_input_idx,
                                        target_actions)
        target_q_values = self.target_critic.predict_on_batch(
            state1_batch_with_action).flatten()
        assert target_q_values.shape == (self.batch_size, )

        # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
        # but only for the affected output units (as given by action_batch).
        discounted_reward_batch = self.gamma * target_q_values
        discounted_reward_batch *= batches.terminal_1
        assert discounted_reward_batch.shape == batches.reward.shape
        targets = (batches.reward + discounted_reward_batch).reshape(
            self.batch_size, 1)

        # Perform a single batch update on the critic network.
        if len(self.critic.inputs) >= 3:
            state0_batch_with_action = batches.state_0[:]
        else:
            state0_batch_with_action = [batches.state_0]
        state0_batch_with_action.insert(self.critic_action_input_idx,
                                        batches.action)
        metrics = self.critic.train_on_batch(state0_batch_with_action, targets)
        if self.processor is not None:
            metrics += self.processor.metrics

        return (metrics)

    def fit_actor(self, batches):
        # Update actor, if warm up is over.
        if self.step > self.nb_steps_warmup_actor:
            # TODO: implement metrics for actor
            if len(self.actor.inputs) >= 2:
                inputs = batches.state_0[:] + batches.state_0[:]

            else:
                inputs = [batches.state_0, +batches.state_0]
            if self.uses_learning_phase:
                inputs += [self.training]
            action_values = self.actor_train_fn(inputs)[0]
            assert action_values.shape == (self.batch_size, self.nb_actions)

    def process_batches(self):
        """
        Process the batches
        Split each a batch of experiences into batches of state_0, action etc...
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
            # FIXME: The keras functions (predict_in_batch) expect to have a list of batches for the states
            # state0_batch = [[e1.state0], [e2.state_0], ...]
            state0_batch.append([e.state0])
            state1_batch.append([e.state1])
            reward_batch.append(e.reward)
            action_batch.append(e.action)
            terminal1_batch.append(0. if e.terminal1 else 1.)

        # Prepare and validate parameters.
        # TODO: Remove this, we do not need processors for now
        state0_batch = self.process_state_batch(state0_batch)
        state1_batch = self.process_state_batch(state1_batch)
        terminal1_batch = np.array(terminal1_batch)
        reward_batch = np.array(reward_batch)
        action_batch = np.array(action_batch)
        assert reward_batch.shape == (self.batch_size, )
        assert terminal1_batch.shape == reward_batch.shape
        assert action_batch.shape == (self.batch_size, self.nb_actions)

        batches = Batches(
            state_0=state0_batch,
            action=action_batch,
            reward=reward_batch,
            terminal_1=terminal1_batch,
            state_1=state1_batch)
        return (batches)


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
