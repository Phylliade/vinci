# For python2 support
from __future__ import print_function
import warnings
from copy import deepcopy

import numpy as np
from keras.callbacks import History
import keras.backend as K

from rl.callbacks import TestLogger, TrainEpisodeLogger, TrainIntervalLogger, Visualizer, CallbackList
from rl.hooks import Hooks
# Other hooks are imported on the fly when required


# Global variables
STEPS_TERMINATION = 1
EPISODES_TERMINATION = 2


class Agent(object):
    """Generic agent class"""

    def __init__(self):
        # Use the same session as Keras
        self.session = K.get_session()
        # self.session = tf.Session()

        # Collected metrics
        self.metrics = {}
        # Internal TF variables
        self.variables = {}
        # And their corresponding summaries
        self.summary_variables = {}

        # Setup hook variables
        self._hook_variables = ["training", "step", "episode", "episode_step", "done", "step_summaries"]
        self._hook_variables_optional = ["reward", "episode_reward", "observation"]
        # Set them to none as default, only if not defined
        for variable in (self._hook_variables + self._hook_variables_optional):
            setattr(self, variable, getattr(self, variable, None))

        # Persistent values
        self.step = 0
        self.episode = 0

        self.checkpoints = []

    def compile(self):
        """Compile an agent: Create the internal variables and populate the variables objects."""
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
             reward_scaling=1.,
             plots=True,
             tensorboard=True):
        """
        Run steps until termination.
        This method shouldn't be called directly, but instead called in :func:`fit` and :func:`test`

        Termination can be either:

        * Maximal number of steps
        * Maximal number of episodes

        :param nb_steps: Number of steps before termination.
        :param nb_episodes: Number of episodes before termination.
        :param bool training: Whether to train or test the agent. Not available for the :func:`fit` and :func:`test` methods.
        :param int action_repetition: Number of times the action is repeated for each step.
        :param callbacks:
        :param int verbose: 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
        :param bool visualize: Render the environment in realtime. This slows down by a big factor (up to 100) the function.
        :param nb_max_start_steps:
        :param start_step_policy: (`lambda observation: action`): The policy to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
        :param log_interval:
        :param reward_scaling:
        :param plots: Plot metrics during training.
        :param tensorboard: Export metrics to tensorboard.
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
                "Please specify one (and only one) of nb_steps or nb_episodes"
            ))
        elif (nb_steps is not None and nb_episodes is None):
            termination_criterion = STEPS_TERMINATION
        elif (nb_steps is None and nb_episodes is not None):
            termination_criterion = EPISODES_TERMINATION
        elif (nb_steps is not None and nb_episodes is not None):
            raise (ValueError(
                "Please specify one (and only one) of nb_steps or nb_episodes"
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
        hooks_list = []
        if tensorboard:
            from rl.hooks.tensorboard import TensorboardHook
            hooks_list.append(TensorboardHook())
        if plots:
            from rl.hooks.plot import PortraitHook, TrajectoryHook
            hooks_list.append(PortraitHook())
            hooks_list.append(TrajectoryHook())
        hooks = Hooks(self, hooks_list)

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
        # Where observation: Observation before the step
        # observation_1: Observation after the step
        self.observation = None
        self.observation_1 = None
        self.action = None
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
                    observation_0 = self.observation_1
                    # Increment the episode step

                # FIXME: Use only one of the two variables
                self.observation = observation_0

                # Increment the current step in both cases
                self.step += 1
                self.episode_step += 1
                self.reward = 0.
                self.step_summaries = []
                accumulated_info = {}

                # Run a single step.
                callbacks.on_step_begin(self.episode_step)
                # This is were all of the work happens. We first perceive and compute the action
                # (forward step) and then use the reward to improve (backward step).

                # state_0 -- (foward) --> action
                self.action = self.forward(self.observation)

                # action -- (step) --> (reward, state_1, terminal)
                # Apply the action
                # With repetition, if necesarry
                for _ in range(action_repetition):
                    callbacks.on_action_begin(self.action)
                    self.observation_1, r, self.done, info = env.step(self.action)
                    # observation_1 = deepcopy(observation_1)

                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    callbacks.on_action_end(self.action)

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
                self.backward()

                # Hooks
                hooks()

                # Callbacks
                # Collect statistics
                step_logs = {
                    'action': self.action,
                    'observation': self.observation_1,
                    'reward': self.reward,
                    # For legacy callbacks upport
                    'metrics': [],
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
            callbacks.on_action_begin(action)
            observation, reward, done, info = env.step(action)
            observation = deepcopy(observation)
            callbacks.on_action_end(action)

            if done:
                warnings.warn(
                    'Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.
                    format(nb_random_start_steps))
                observation = deepcopy(env.reset())
                break
        return (observation)

    def fit(self, **kwargs):
        """
        Train the agent. On the contrary of :func:`test`, learning is involved
        See :func:`_run` for the argument list.
        """
        return(self._run(training=True, **kwargs))

    def test(self, **kwargs):
        """
        Test the agent. On the contrary of :func:`fit`, no learning is involved
        Also, noise is removed.
        """
        return(self._run(training=False, **kwargs))

    def fit_offline(self,
                    epochs=1,
                    episode_length=20,
                    plots=True,
                    tensorboard=True,
                    **kwargs):
        """Train the networks in offline mode"""

        self.training = True
        self.done = True

        # Initialize the Hooks
        hooks_list = []
        if tensorboard:
            from rl.hooks.tensorboard import TensorboardHook
            hooks_list.append(TensorboardHook())
        if plots:
            from rl.hooks.plot import PortraitHook, TrajectoryHook
            hooks_list.append(PortraitHook())
            hooks_list.append(TrajectoryHook())
        hooks = Hooks(self, hooks_list)

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
            self.step_summaries = []

            # Finish the step
            if (epoch % episode_length == 0):
                self.done = True

            # Post step
            # Train the networks
            print("\rTraining epoch: {}/{} ".format(epoch + 1, epochs), end="")
            self.backward(offline=True)

            # Hooks
            hooks()

    def reset_states(self):
        """Resets all internally kept states after an episode is completed."""
        pass

    def forward(self, observation):
        """
        Takes the an observation from the environment and returns the action to be taken next.
        If the policy is implemented by a neural network, this corresponds to a forward (inference) pass.

        :param observation: The observation from which we want an action
        :return: The desired action
        """
        raise NotImplementedError()

    def backward(self, **kwargs):
        """
        Train the agent controllers by using the training strategy.
        In general, backward is a wrapper around fit_controllers: It selects which controlers to train (e.g. actor, critic, memory)
        """
        raise NotImplementedError()

    def fit_controllers(self, **kwargs):
        """
        Train the agent controllers

        This is an internal method used to directly train the controllers. The learning strategy is defined by :func:`backward`.
        """
        raise(NotImplementedError())

    def load_weights(self, filepath):
        """
        Loads the weights of an agent from an HDF5 file.

        :param str filepath: The path to the HDF5 file.
        """
        raise NotImplementedError()

    def save_weights(self, filepath, overwrite=False):
        """
        Saves the weights of an agent as an HDF5 file.

        :param str filepath: The path to where the weights should be saved.
        :param bool overwrite: If `False` and `filepath` already exists, raises an error.
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
