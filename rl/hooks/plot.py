from .hooks import Hook
from rl.utils.plot import portrait_critic, portrait_actor, plot_trajectory, plot_distribution, plot_action_distribution


class PortraitHook(Hook):
    # TODO: Move this hook to a non-blocking thread
    def __call__(self):
        if self.agent.done:

            # Get the file name
            if self.agent.training:
                file_name = "{}.png"
            else:
                file_name = "test/{}.png"
            file_name = file_name.format(self.count)

            # Only plot portraits for envs whose observation_space is 2-dimensional
            if self.agent.env.observation_space.dim == 2:
                # Plot the phase portrait of the actor and the critic
                portrait_actor(
                    self.agent.actor,
                    self.agent.env,
                    save_figure=True,
                    figure_file=("figures/actor/" + file_name))
                portrait_critic(
                    self.agent.critic,
                    self.agent.env,
                    save_figure=True,
                    figure_file=("figures/critic/" + file_name))

            # Plot the distribution of the actor and the critic
            plot_distribution(
                self.agent.actor,
                self.agent.critic,
                self.agent.env,
                actor_file="figures/actor/distribution/{}.png".format(
                    self.count),
                critic_file="figures/critic/distribution/{}.png".format(
                    self.count))


class TrajectoryHook(Hook):
    """Records the trajectory of the agent"""

    def __init__(self, *args, **kwargs):
        super(TrajectoryHook, self).__init__(*args, **kwargs)
        self.trajectory = {"x": [], "y": []}

    def __call__(self):
        # Only plot portraits for envs whose observation_space is 2-dimensional
        if self.agent.env.observation_space.dim == 2:
            self.trajectory["x"].append(self.agent.observation[0])
            self.trajectory["y"].append(self.agent.observation[1])

            if self.agent.done:
                plot_trajectory(
                    self.trajectory,
                    self.agent.actor,
                    self.agent.env,
                    figure_file=("figures/trajectory/{}.png".format(self.count)))
                # Flush the trajectories
                self.trajectory["x"] = []
                self.trajectory["y"] = []


class MemoryDistributionHook(Hook):
    def __call__(self):
        if self.agent.done:
            replay_buffer = self.agent.memory.dump()
            # Collect every states and actions
            actions = []
            states = []

            for experience in replay_buffer:
                actions.append(experience.action)
                states.append(experience.state0)

            plot_action_distribution(actions, file="figures/memory/action/{}.png".format(self.count))
