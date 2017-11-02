import numpy as np
from rl.runtime.agent import Agent
from rl.utils.printer import print_epoch, print_info
from rl.runtime.experiment import DefaultExperiment


class GEPAgent(Agent):
    def __init__(self, environment, model, im_model, experiment=None, **kwargs):
        self.environment = environment
        self.model = model
        self.im_model = im_model

        if experiment is None:
            # Create the experiment ourselves but we are losing the possibility to use it as a context manager
            self.experiment = DefaultExperiment(use_tf=False)
        else:
            self.experiment = experiment

        # Finish with agent initialization since Hook initialization can depend on custom GEPAgent variables
        super(GEPAgent, self).__init__(experiment=self.experiment, **kwargs)

    def _run(self,
             n_episodes,
             train=True,
             goal=None,
             save_to_replay_buffer=False,
             noisy_action=False,
             noise_intensity=0.3,
             render=False,
             noisy_policy_parameters=True,
             verbosity=False):
        # Configure hook variables
        self.training = train
        if noisy_policy_parameters:
            self.model.mode = "explore"
        else:
            self.model.mode = "exploit"

        self.done = False

        # Select goal if defined for entire episode
        episodic_goal = (goal is None)

        if not episodic_goal:
            self.goal = goal

        self.hooks.run_init()

        for epoch in range(1, n_episodes + 1):
            print_epoch(epoch, n_episodes)
            self.hooks.episode_init()
            self.step += self.environment.rollout_size
            self.episode_step = self.environment.rollout_size
            self.step_summaries = []
            if self.training:
                self.training_step += self.environment.rollout_size
            self.episode += 1
            if self.training:
                self.training_episode += 1

            # Select goal
            if episodic_goal:
                self.goal = self.im_model.sample()

            if verbosity:
                print("Running on goal: {}".format(self.goal))

            # Invert it
            self.policy = self.model.inverse_prediction(self.goal)

            # Compute the achievement and store it in the database
            (self.achievement,
             self.episode_reward) = self.environment.compute_sensori_effect(
                 self.policy,
                 save_to_replay_buffer=save_to_replay_buffer,
                 noisy_action=noisy_action,
                 noise_intensity=noise_intensity,
                 render=render,
                 hooks=self.hooks)

            if verbosity:
                print("achievement: {}".format(self.achievement))

            # Backward
            # TODO: Move this to a backward method
            if train:
                self.model.update(self.policy, self.achievement)

            self.done = True

            # End of episode
            # Compute error
            self.error = np.linalg.norm(self.goal - self.achievement)

            # Hooks
            self.hooks.episode_end()
        self.hooks.run_end()

    def test(self,
             n_episodes,
             noisy_policy_parameters=False,
             goal=None,
             **kwargs):
        if goal is None:
            str = "Beginning a test of {} episodes with random goals".format(
                n_episodes)
        else:
            str = "Beginning a test of {} episodes on the goal {}".format(
                n_episodes, goal)
        print_info(str)
        return (self._run(
            n_episodes,
            train=False,
            goal=goal,
            noisy_policy_parameters=noisy_policy_parameters,
            **kwargs))

    def train(self,
              n_episodes,
              noisy_policy_parameters=True,
              goal=None,
              **kwargs):
        if goal is None:
            str = "Training for {} episodes with random goals".format(
                n_episodes)
        else:
            str = "Training for {} episodes on the goal {}".format(
                n_episodes, goal)
        print_info(str)
        return (self._run(
            n_episodes,
            train=True,
            goal=goal,
            noisy_policy_parameters=noisy_policy_parameters,
            **kwargs))

    def test_goal(self, goal):
        pass

    def bootstrap(self,
                  n_bootstrap,
                  save_to_replay_buffer=False,
                  noisy_action=False,
                  render=False):
        print_info("Bootstrapping: Motor babbling...")
        if n_bootstrap <= 0:
            raise (ValueError("0 bootstrap epoch selected, select at least 1"))

        for epoch, m in enumerate(
                self.environment.random_motors(n=n_bootstrap)):
            s, _ = self.environment.compute_sensori_effect(
                m,
                save_to_replay_buffer=save_to_replay_buffer,
                noisy_action=noisy_action,
                render=render,
                hooks=self.hooks)
            print("Iteration {}/{}. Achievement: {}".format(
                epoch + 1, n_bootstrap, s))
            self.model.update(m, s)
