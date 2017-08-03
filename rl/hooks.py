from rl.utils.plot import portrait_critic, portrait_actor


class Hook:
    def __init__(self, agent):
        """
        The abstract Hook class
        :param agent: the RL agent
        """
        self.agent = agent

    def __call__(self):
        """
        Generic hook.
        Better if it doesn't return anything and instead exports the data itself (e.g. pickle, image)
        """
        raise(NotImplementedError)

    @property
    def count(self):
        return(self.agent.episode)


class PortraitHook(Hook):
    # TODO: Move this hook to a non-blocking thread
    def __call__(self):
        file_name = "portrait_{}.png".format(self.count)
        portrait_actor(self.agent.actor, save_figure=True, figure_file=("figures/actor/" + file_name))
        portrait_critic(self.agent.critic, save_figure=True, figure_file=("figures/critic/" + file_name))


class TrajectoryHook(Hook):
    """Records the trajectory of the agent"""
    def __call__(self):
        pass
