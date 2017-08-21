from rl.experiment import Experiments
from rl.agents.ddpg import DDPGAgent
from rl.memory import SimpleMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.utils.networks import simple_actor, simple_critic
import gym
from rl.utils.env import populate_env

my_expe = Experiments("test")

for experiment in my_expe(5):
    # Get the environment
    # And populate it with useful metadata
    env = populate_env(gym.make("MountainCarContinuous-v0"))

    # Build the actor and the critic
    actor = simple_actor(env)
    critic = simple_critic(env)

    # Memory
    memory = SimpleMemory(env=env, limit=1000000)

    # Noise
    random_process = OrnsteinUhlenbeckProcess(
        size=env.action_space.dim, theta=.15, mu=0., sigma=3.)

    # Agent
    agent = DDPGAgent(
        actor=actor,
        critic=critic,
        env=env,
        memory=memory,
        random_process=random_process,
        experiment=experiment,
    )
    agent.compile()

    agent.fit(
        env=env,
        nb_episodes=1,
        visualize=False,
        verbose=2,
        nb_max_episode_steps=200,
        plots=False)
