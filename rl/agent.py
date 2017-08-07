# For python2 support
from __future__ import print_function
import warnings
from copy import deepcopy

import numpy as np
from keras.callbacks import History
import keras.backend as K

from rl.callbacks import TestLogger, TrainEpisodeLogger, TrainIntervalLogger, Visualizer, CallbackList
from rl.hooks import PortraitHook, TensorboardHook, TrajectoryHook, Hooks
from rl.core import Processor


# Global variables
STEPS_TERMINATION = 1
EPISODES_TERMINATION = 2


class Agent(object):
    """Abstract base class for all implemented agents.

    Each agent interacts with the environment (as defined by the `Env` class) by first observing the
    state of the environment. Based on this observation the agent changes the environment by performing
    an action.

    Do not use this abstract base class directly but instead use one of the concrete agents implemented.
    Each agent realizes a reinforcement learning algorithm. Since all agents conform to the same
    interface, you can use them interchangeably.

    To implement your own agent, you have to implement the following methods:

    - `forward`
    - `backward`
    - `compile`
    - `load_weights`
    - `save_weights`
    - `layers`

    # Arguments
        processor (`Processor` instance): See [Processor](#processor) for details.
    """

    def __init__(self, processor=None):
        # Use a default identity processor if not provided
        if processor is None:
            self.processor = Processor()

        # Use the same session as Keras
        self.session = K.get_session()
        # self.session = tf.Session()

        # Setup hook variables
        self._hook_variables = ["training", "step", "episode", "episode_step", "done", "step_summaries"]
        self._hook_variables_optional = ["reward", "episode_reward", "observation"]
        # Set them to none as default, only if not defined
        for variable in (self._hook_variables + self._hook_variables_optional):
            setattr(self, variable, getattr(self, variable, None))

        # Persistent values
        self.step = 0
        self.episode = 0

    def compile(self, optimizer, metrics=[]):
        """Compiles an agent and the underlaying models to be used for training and testing.

        # Arguments
            optimizer (`keras.optimizers.Optimizer` instance): The optimizer to be used during training.
            metrics (list of functions `lambda y_true, y_pred: metric`): The metrics to run during training.
        """
        raise NotImplementedError()

    def _run(self,
             env,
             nb_steps=None,
             nb_episodes=None,
             training=True,
             action_repetition=1,
             callbacks=None,
             verbose=1,
             visualize=False,
             nb_max_start_steps=0,
             start_step_policy=None,
             log_interval=10000,
             nb_max_episode_steps=None,
             reward_scaling=1.):
        """Trains the agent on the given environment.

        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_steps (integer): Number of training steps to be performed.
            nb_episodes (integer): Number of episodes to perform
            training (boolean): Whether to train or test the agent
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.
            reward_scaling (float): The amount with which the reward will be scaled

        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        if not self.compiled:
            raise RuntimeError(
                'Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.'
            )
        if action_repetition < 1:
            raise ValueError('action_repetition must be >= 1, is {}'.format(
                action_repetition))

        # Process the different cases when either nb_steps or nb_episodes are specified
        if (nb_steps is None and nb_episodes is None):
            raise (ValueError(
                "Please specify one (and only one) of nb_steps and nb_episodes"
            ))
        elif (nb_steps is not None and nb_episodes is None):
            termination_criterion = STEPS_TERMINATION
        elif (nb_steps is None and nb_episodes is not None):
            termination_criterion = EPISODES_TERMINATION
        elif (nb_steps is not None and nb_episodes is not None):
            raise (ValueError(
                "Please specify one (and only one) of nb_steps and nb_episodes"
            ))

        self.training = training

        # Initialize callbacks
        if callbacks is None:
            callbacks = []
        if self.training:
            if verbose == 1:
                callbacks += [TrainIntervalLogger(interval=log_interval)]
            elif verbose > 1:
                callbacks += [TrainEpisodeLogger()]
        else:
            if verbose >= 1:
                callbacks += [TestLogger()]
        callbacks = [] if not callbacks else callbacks[:]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)

        if termination_criterion == STEPS_TERMINATION:
            params = {
                'nb_steps': nb_steps,
            }
        elif termination_criterion == EPISODES_TERMINATION:
            params = {
                'nb_episodes': nb_episodes,
                'nb_steps': 1,
            }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)

        # Initialize the Hooks
        hooks = Hooks(self, [TensorboardHook(), PortraitHook(), TrajectoryHook()])

        # Define the termination criterion
        # Step and episode at which we satrt the function
        start_step = self.step
        start_episode = self.episode
        if termination_criterion == STEPS_TERMINATION:
            def termination():
                return (self.step - start_step > nb_steps)
        elif termination_criterion == EPISODES_TERMINATION:
            def termination():
                return (self.episode - start_episode > nb_episodes)

        if self.training:
            self._on_train_begin()
        else:
            self._on_test_begin()

        callbacks.on_train_begin()

        # Setup
        self.done = True
        did_abort = False
        # Define these for clarification, not mandatory:
        # Where observation_0: Observation before the step
        # observation_1: Observation after the step
        observation_0 = None
        observation_1 = None
        self.step_summaries = None

        try:
            # Run steps (and episodes) until the termination criterion is met
            while not (termination()):
                # Init episode
                # If we are at the beginning of a new episode, execute a startup sequence
                if self.done:
                    self.episode += 1
                    self.episode_reward = 0.
                    self.episode_step = 0
                    callbacks.on_episode_begin(self.episode)

                    # Obtain the initial observation by resetting the environment.
                    self.reset_states()
                    observation_0 = deepcopy(env.reset())
                    assert observation_0 is not None

                    # Perform random steps at beginning of episode and do not record them into the experience.
                    # This slightly changes the start position between games.
                    if nb_max_start_steps != 0:
                        observation_0 = self._perform_random_steps(
                            nb_max_start_steps, start_step_policy, env,
                            observation_0, callbacks)

                else:
                    # We are in the middle of an episode
                    # Update the observation
                    observation_0 = observation_1
                    # Increment the episode step

                # FIXME: Use only one of the two variables
                self.observation = observation_0

                # Increment the current step in both cases
                self.step += 1
                self.episode_step += 1
                self.reward = 0.
                accumulated_info = {}

                # Run a single step.
                callbacks.on_step_begin(self.episode_step)
                # This is were all of the work happens. We first perceive and compute the action
                # (forward step) and then use the reward to improve (backward step).

                # state_0 -- (foward) --> action
                action = self.forward(observation_0)
                # Process the action
                action = self.processor.process_action(action)

                # action -- (step) --> (reward, state_1, terminal)
                # Apply the action
                # With repetition, if necesarry
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation_1, r, self.done, info = env.step(action)
                    # observation_1 = deepcopy(observation_1)

                    observation_1, r, self.done, info = self.processor.process_step(
                        observation_1, r, self.done, info)
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    callbacks.on_action_end(action)

                    self.reward += r

                    # Set episode as finished if the environment has terminated
                    if self.done:
                        break

                # Scale the reward
                self.reward = self.reward * reward_scaling
                self.episode_reward += self.reward

                # End of the step
                # Stop episode if reached the step limit
                if nb_max_episode_steps and self.episode_step >= nb_max_episode_steps:
                    # Force a terminal state.
                    self.done = True

                # Post step: training, callbacks and hooks
                # Train the algorithm
                metrics, self.step_summaries = self.backward(
                    observation_0,
                    action,
                    self.reward,
                    observation_1,
                    terminal=self.done)

                # Hooks
                hooks()

                # Callbacks
                # Collect statistics
                step_logs = {
                    'action': action,
                    'observation': observation_1,
                    'reward': self.reward,
                    'metrics': metrics,
                    'episode': self.episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(self.episode_step, step_logs)

                # Episodic callbacks
                if self.done:
                    # Collect statistics
                    episode_logs = {
                        'episode_reward': np.float_(self.episode_reward),
                        'nb_episode_steps': np.float_(self.episode_step),
                        'nb_steps': np.float_(self.step),
                    }
                    callbacks.on_episode_end(self.episode, logs=episode_logs)

        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that
            # the `on_train_end` method is properly called.
            did_abort = True

        callbacks.on_train_end(logs={'did_abort': did_abort})
        self._on_train_end()

        return(history)

    def _perform_random_steps(self, nb_max_start_steps, start_step_policy, env,
                              observation, callbacks):
        nb_random_start_steps = np.random.randint(nb_max_start_steps)
        for _ in range(nb_random_start_steps):
            if start_step_policy is None:
                action = env.action_space.sample()
            else:
                action = start_step_policy(observation)
            action = self.processor.process_action(action)
            callbacks.on_action_begin(action)
            observation, reward, done, info = env.step(action)
            observation = deepcopy(observation)
            observation, reward, done, info = self.processor.process_step(
                observation, reward, done, info)
            callbacks.on_action_end(action)

            if done:
                warnings.warn(
                    'Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.
                    format(nb_random_start_steps))
                observation = deepcopy(env.reset())
                observation = self.processor.process_observation(observation)
                break
        return (observation)

    def fit(self, **kwargs):
        """
        Train the agent on the given environment.

        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_steps (integer): Number of training steps to be performed.
            or nb_episodes
        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        return(self._run(training=True, **kwargs))

    def test(self, **kwargs):
        """
        Test the agent on the given environment.
        In training mode, noise is removed.
        """
        return(self._run(training=False, **kwargs))

    def fit_offline(self,
                    fit_critic=True,
                    fit_actor=True,
                    hard_update_target_critic=False,
                    hard_update_target_actor=False,
                    epochs=1,
                    episode_length=20):
        """Train the networks in offline mode"""

        self.training = True
        self.done = True

        hooks = Hooks(self, [TensorboardHook(), PortraitHook()])

        # We could use a termination criterion, based on step instead of epoch, as in  _run
        for epoch in range(epochs):
            if self.done:
                self.episode += 1
                self.episode_reward = 0.
                self.episode_step = 0

            # Initialize the step
            self.done = False
            self.step += 1
            self.episode_step += 1

            print("\rTraining epoch: {} ".format(epoch), end="")
            self.step_summaries = self.fit_controllers(
                fit_critic=fit_critic,
                fit_actor=fit_actor,
                hard_update_target_critic=hard_update_target_critic,
                hard_update_target_actor=hard_update_target_actor)

            if (epoch % episode_length == 0):
                self.done = True

            # Post step
            hooks()

    def reset_states(self):
        """Resets all internally kept states after an episode is completed."""
        pass

    def forward(self, observation):
        """Takes the an observation from the environment and returns the action to be taken next.
        If the policy is implemented by a neural network, this corresponds to a forward (inference) pass.

        # Argument
            observation (object): The current observation from the environment.

        # Returns
            The next action to be executed in the environment.
        """
        raise NotImplementedError()

    def backward(self, observation_0, action, reward, observation_1, terminal):
        """
        Train the agent controller according to the training strategy.

        # Argument
            reward (float): The observed reward after executing the action returned by `forward`.
            terminal (boolean): `True` if the new state of the environment is terminal.
        """
        raise NotImplementedError()

    def fit_controllers(self):
        """
        Train the agent controllers

        This is an internal method used to directly train the controllers. There is no learning strategy.
        It should be used by backward.
        """
        raise(NotImplementedError())

    def load_weights(self, filepath):
        """Loads the weights of an agent from an HDF5 file.

        # Arguments
            filepath (str): The path to the HDF5 file.
        """
        raise NotImplementedError()

    def save_weights(self, filepath, overwrite=False):
        """Saves the weights of an agent as an HDF5 file.

        # Arguments
            filepath (str): The path to where the weights should be saved.
            overwrite (boolean): If `False` and `filepath` already exists, raises an error.
        """
        raise NotImplementedError()

    def get_config(self):
        """Configuration of the agent for serialization.
        """
        return {}

    @property
    def layers(self):
        """Returns all layers of the underlying model(s).

        If the concrete implementation uses multiple internal models,
        this method returns them in a concatenated list.
        """
        raise NotImplementedError()

    @property
    def metrics_names(self):
        """The human-readable names of the agent's metrics. Must return as many names as there
        are metrics (see also `compile`).
        """
        return []

    def _on_train_begin(self):
        """Callback that is called before training begins."
        """
        pass

    def _on_train_end(self):
        """Callback that is called after training ends."
        """
        pass

    def _on_test_begin(self):
        """Callback that is called before testing begins."
        """
        pass

    def _on_test_end(self):
        """Callback that is called after testing ends."
        """
        pass
