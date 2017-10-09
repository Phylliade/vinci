from .hook import Hook
from rl.utils.plot import portrait_critic, portrait_actor, plot_trajectory, plot_distribution, plot_action_distribution


class PortraitHook(Hook):
    def agent_init(self, *args, **kwargs):
        super(PortraitHook, self).agent_init(*args, **kwargs)
        # Endpoints
        # Portrait
        self.endpoint_actor = self.experiment.endpoint("figures/portrait/actor")
        self.endpoint_actor_test = self.experiment.endpoint("figures/portrait/actor/test")
        self.endpoint_critic = self.experiment.endpoint("figures/portrait/critic")
        self.endpoint_critic_test = self.experiment.endpoint("figures/portrait/critic/test")
        # Distribution
        self.endpoint_actor_distribution = self.experiment.endpoint("figures/distribution/actor")
        self.endpoint_critic_distribution = self.experiment.endpoint("figures/distribution/critic")

    def episode_end(self):
        print("Plotting portrait")
        # Get the file name
        if not self.agent.training:
            actor_endpoint = self.endpoint_actor_test
            critic_endpoint = self.endpoint_critic_test
        else:
            actor_endpoint = self.endpoint_actor
            critic_endpoint = self.endpoint_critic

        file_name = "{}.png".format(self.count)

        # Only plot portraits for envs whose observation_space is 2-dimensional
        if self.agent.env.observation_space.dim == 2:
            # Plot the phase portrait of the actor and the critic
            portrait_actor(
                self.agent.actor,
                self.agent.env,
                save_figure=True,
                figure_file=(actor_endpoint + file_name))
            portrait_critic(
                self.agent.critic,
                self.agent.env,
                save_figure=True,
                figure_file=(critic_endpoint + file_name))

        # Plot the distribution of the actor and the critic
        plot_distribution(
            self.agent.actor,
            self.agent.critic,
            self.agent.env,
            actor_file=(self.endpoint_actor_distribution + file_name),
            critic_file=(self.endpoint_actor_distribution + file_name))


class TrajectoryHook(Hook):
    """Records the trajectory of the agent"""
    def agent_init(self, *args, **kwargs):
        super(TrajectoryHook, self).agent_init(*args, **kwargs)
        self.trajectory = {"x": [], "y": []}
        self.endpoint = self.experiment.endpoint("figures/trajectory")

    def step_end(self):
        # Only plot portraits for envs whose observation_space is 2-dimensional
        if self.agent.env.observation_space.dim == 2:
            self.trajectory["x"].append(self.agent.observation[0])
            self.trajectory["y"].append(self.agent.observation[1])

    def episode_end(self):
        print("Plotting trajectory")
        plot_trajectory(
            self.trajectory,
            self.agent.actor,
            self.agent.env,
            figure_file=(self.endpoint + "{}.png".format(self.count)))
        # Flush the trajectories
        self.trajectory["x"] = []
        self.trajectory["y"] = []


class MemoryDistributionHook(Hook):
    def agent_init(self, *args, **kwargs):
        super(MemoryDistributionHook, self).agent_init(*args, **kwargs)
        self.endpoint = self.experiment.endpoint("figures/memory/action")

    def episode_end(self):
        print("Plotting memory distribution")
        replay_buffer = self.agent.memory.dump()
        # Collect every states and actions
        actions = []
        states = []

        for experience in replay_buffer:
            actions.append(experience.action)
            states.append(experience.state0)

        plot_action_distribution(actions, file=(self.endpoint + "{}.png".format(self.count)))
