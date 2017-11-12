# For python2 support
import warnings
from copy import deepcopy

import numpy as np
from keras.callbacks import History

from rl.logger.callbacks import Visualizer, CallbackList
from rl.logger.testlogger import TestLogger
from rl.logger.trainepisodelogger import TrainEpisodeLogger
from rl.logger.trainintervallogger import TrainIntervalLogger
from rl.utils.printer import print_status
from rl.runtime.agent import Agent
from rl.utils.numerics import normalize
# Other hooks are imported on the fly when required

# Global variables
STEPS_TERMINATION = 1
EPISODES_TERMINATION = 2

class RLAgent(Agent):
    """Generic agent class"""

    def __init__(self,
                 normalize_observations=False,
                 reward_scaling=1.,
                 **kwargs):
        super(RLAgent, self).__init__(**kwargs)
        self.normalize_observations = normalize_observations
        self.reward_scaling = reward_scaling
        self.abort = False

        # Collected metrics
        self.metrics = {}
        # Internal TF variables
        self.variables = {}
        # And their corresponding summaries
        self.summary_variables = {}

        self.checkpoints = []

    def compile(self):
        """Compile an agent: Create the internal variables and populate the variables objects."""
        raise NotImplementedError()

    def check_run_params(self, action_repetition, max_steps, episodes):
        if not self.compiled:
            raise RuntimeError('Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `train()`.')
        if action_repetition < 1:
            raise ValueError('action_repetition must be >= 1, is {}'.format(action_repetition))

        # Process the different cases when either steps or episodes are specified
        if (max_steps is None and episodes is None):
            raise (ValueError("No duration specified: Please specify one of max_steps or episodes"))
        elif (max_steps is not None and episodes is None):
            termination_criterion = STEPS_TERMINATION
        elif (max_steps is None and episodes is not None):
            termination_criterion = EPISODES_TERMINATION
        elif (max_steps is not None and episodes is not None):
            raise (ValueError('Please specify only one of max_steps or episodes, instead of {} and {}'.format(max_steps, episodes)))
        return termination_criterion

    def init_callbacks(self, callbacks, verbosity, max_steps, episodes, log_interval, render, termination_criterion):
        if callbacks is None:
            callbacks = []
        if self.training:
            if verbosity == 1:
                callbacks += [TrainIntervalLogger(interval=log_interval)]
            elif verbosity > 1:
                callbacks += [TrainEpisodeLogger()]
        else:
            if verbosity == 1:
                callbacks += [TestLogger()]
            elif verbosity > 1:
                callbacks += [TrainEpisodeLogger()]
        callbacks = [] if not callbacks else callbacks[:]
        if render:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(self.env)
        if termination_criterion == STEPS_TERMINATION:
            params = {
                'steps': max_steps,
            }
        elif termination_criterion == EPISODES_TERMINATION:
            params = {
                'episodes': episodes,
                'steps': 1,
            }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        return callbacks, history


    def take_step(self, callbacks, action_repetition):
        # action -- (step) --> (reward, state_1, terminal)
        # Apply the action
        # With repetition, if necessary
        self.step_reward = 0.
        accumulated_info = {}

        print("obs", self.observation_1)
        print("action", self.action)

        for _ in range(action_repetition):
            callbacks.on_action_begin(self.action)
            self.observation_1, reward, self.done, info = self.env.step(self.action)
            if self.normalize_observations:
                self.observation_1 = normalize(self.observation_1)

            for key, value in info.items():
                if not np.isreal(value):
                    continue
                if key not in accumulated_info:
                    accumulated_info[key] = np.zeros_like(value)
                accumulated_info[key] += value

            callbacks.on_action_end(self.action)

            self.step_reward += reward

            print("reward", self.step_reward)

            # Set episode as finished if the self.environment has terminated
            if self.done:
                break

            # Scale the reward
        self.episode_reward += self.step_reward * self.reward_scaling
        return accumulated_info

    def global_init(self):
        # Setup
        self.termination = False
        self.done = False
        # Define these for clarification, not mandatory:
        # Where observation: Observation before the step
        # observation_1: Observation after the step
        self.observation = None
        self.observation_1 = None
        self.action = None
        self.episode_step = 0
        self.step_summaries = None
        self.step_summaries_post = None
        self.hooks.run_init()

    def init_new_episode(self):
        self.episode += 1
        if self.training:
            self.training_episode += 1
        self.episode_reward = 0.
        self.episode_step = 0

    def init_observation(self):
        # Obtain the initial observation by resetting the self.environment.
        self.reset_states()
        self.observation = deepcopy(self.env.reset())
        if self.normalize_observations:
            self.observation = normalize(self.observation)
        assert self.observation is not None

    def incr_step_counters(self):
        self.step += 1
        self.episode_step += 1
        if self.training:
            self.training_step += 1

    def call_step_end(self, callbacks, accumulated_info):
        # End of step Callbacks
        step_logs = {
            'action': self.action,
            'observation': self.observation_1,
            'reward': self.step_reward,
            # For legacy callbacks support
            'metrics': [],
            'episode': self.episode,
            'info': accumulated_info,
        }
        callbacks.on_step_end(self.episode_step, step_logs)

    def call_episode_end(self, callbacks):
        # Collect statistics
        episode_logs = {
            'episode_reward': np.float_(self.episode_reward),
            'nb_episode_steps': np.float_(self.episode_step),
            'steps': np.float_(self.step),
        }
        callbacks.on_episode_end(self.episode, logs=episode_logs)
        self.hooks.episode_end()

    def _run(self,
             max_steps=None,
             episodes=None,
             train=True,
             render=False,
             exploration=True,
             plots=False,
             tensorboard=False,
             callbacks=None,
             verbosity=2,
             action_repetition=1,
             nb_max_episode_steps=None,
             log_interval=10000,
             **kwargs):
        """
        Run steps until termination.
        This method shouldn't be called directly, but instead called in :func:`fit` and :func:`test`

        Termination can be either:

        * Maximal number of steps
        * Maximal number of episodes

        :param max_steps: Number of steps before termination.
        :param episodes: Number of episodes before termination.
        :param bool training: Whether to train or test the agent. Not available for the :func:`fit` and :func:`test` methods.
        :param int action_repetition: Number of times the action is repeated for each step.
        :param callbacks:
        :param int verbosity: 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
        :param bool render: Render the self.environment in realtime. This slows down by a big factor (up to 100) the function.
        :param log_interval:
        :param reward_scaling:
        :param plots: Plot metrics during training.
        :param tensorboard: Export metrics to tensorboard.
        """
        termination_criterion = self.check_run_params(action_repetition, max_steps, episodes)

        self.training = train
        # We explore only if the flag is selected and we are in train mode
        self.exploration = (train and exploration)

        # Initialize callbacks
        callbacks, history = self.init_callbacks(callbacks, verbosity, max_steps, episodes, log_interval, render, termination_criterion)
 
        # Add run hooks
        if tensorboard:
            from rl.hooks.tensorboard import TensorboardHook
            self.hooks.append(TensorboardHook(agent_id=self.id))
        if plots:
            from rl.hooks.plot import PortraitHook, TrajectoryHook
            self.hooks.append(PortraitHook(agent_id=self.id))
            self.hooks.append(TrajectoryHook(agent_id=self.id))

        # Define the termination criterion
        # Step and episode at which we start the function
        start_step = self.step
        start_episode = self.episode

        if self.training:
            self._on_train_begin()
        else:
            self._on_test_begin()

        callbacks.on_train_begin()

        # Init episode
        self.global_init()
        self.init_new_episode()
        callbacks.on_episode_begin(self.episode)
        self.init_observation()


        # Run steps (and episodes) until the termination criterion is met
        while not (self.termination):
            if termination_criterion == STEPS_TERMINATION:
                self.termination = (self.step - start_step > max_steps)
            elif termination_criterion == EPISODES_TERMINATION:
                self.termination = (self.episode - start_episode > episodes)
            else:
                print("termination criterion undefined")

            # Increment the current step in both cases
            self.incr_step_counters()

            self.step_summaries = []
            self.step_summaries_post = []

            # Run a single step.
            callbacks.on_step_begin(self.episode_step)

            # This is were all of the work happens. We first perceive and compute the action
            # (forward step) and then use the reward to improve (backward step).

            # state_0 -- (forward) --> action
            print("obs pre : ", self.observation)
            self.action = self.forward(self.observation)
            print("action pre : ", self.action)

            accumulated_info = self.take_step(callbacks, action_repetition)

            # End of the step
            # Stop episode if reached the step limit
            if nb_max_episode_steps and self.episode_step >= nb_max_episode_steps:
                self.done = True

            # Post step: training, callbacks and hooks
            # Train the algorithm
            self.backward()

            self.hooks()

            # step_end Hooks
            self.call_step_end(callbacks, accumulated_info)

            # Stop run if termination criterion met
            if self.abort or self.done:
                #print("episode end step : ", self.step)
                # End of episode callbacks
                self.call_episode_end(callbacks)

            self.termination = self.termination or self.abort

            # If we are at the beginning of a new episode, execute a startup sequence
            if self.done:
                self.init_new_episode()
                callbacks.on_episode_begin(self.episode)
                self.init_observation()
            else:
                # We are in the middle of an episode
                # Update the observation
                self.observation = self.observation_1


        callbacks.on_train_end(logs={'did_abort': self.abort})
        self._on_train_end()
        self.hooks.run_end()

        return (history)

    def train_offline(self,
                      max_steps=1,
                      episode_length=200,
                      plots=False,
                      tensorboard=False,
                      verbosity=True,
                      **kwargs):
        """Train the networks in offline mode, from the replay buffer, that is without calling upon the environment"""

        self.training = True
        self.done = True

        # Add run hooks
        if tensorboard:
            from rl.hooks.tensorboard import TensorboardHook
            self.hooks.append(TensorboardHook(agent_id=self.id))
        if plots:
            from rl.hooks.plot import PortraitHook
            self.hooks.append(PortraitHook(agent_id=self.id))

        # Run_init hooks
        self.hooks.run_init()

        # We could use a termination criterion, based on step instead of epoch, as in  _run
        for epoch in range(1, max_steps+1):
            if self.done:
                self.init_new_episode()

            # Initialize the step
            self.done = False
            self.incr_step_counters()

            self.step_summaries = []
            self.step_summaries_post = []

            # Finish the step
            if (epoch % episode_length == 0):
                self.done = True

            # Post step
            # Train the networks
            if verbosity:
                print_status("Training epoch: {}/{} ".format(epoch, max_steps),terminal=(epoch == max_steps))
            self.backward_offline(**kwargs)

            # Hooks
            self.hooks()

            if self.done:
                self.hooks.episode_end()

            if self.abort:
                break

        # End of the run
        self.hooks.run_end()

    def reset_states(self):
        """Resets all internally kept states after an episode is completed."""
        pass

    def forward(self, observation):
        """
        Takes the an observation from the self.environment and returns the action to be taken next.
        If the policy is implemented by a neural network, this corresponds to a forward (inference) pass.

        :param observation: The observation from which we want an action
        :return: The desired action
        """
        raise NotImplementedError()

    def backward(self, **kwargs):
        """
        Train the agent controllers by using the training strategy.
        In general, backward is a wrapper around train_controllers: It selects which controlers to train (e.g. actor, critic, memory)
        """
        raise NotImplementedError()

    def train_controllers(self, **kwargs):
        """
        Train the agent controllers

        This is an internal method used to directly train the controllers. The learning strategy is defined by :func:`backward`.
        """
        raise (NotImplementedError())

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
