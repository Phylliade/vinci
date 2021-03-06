from rl.agents.ddpg import DDPGAgent
from rl.memory import SimpleMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.utils.networks import simple_actor, simple_critic
import gym
from rl.utils.env import populate_env
from rl.runtime.experiment import DefaultExperiment

# Experiment
experiment = DefaultExperiment()

with experiment:
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
        experiment=experiment,
        actor=actor,
        critic=critic,
        env=env,
        memory=memory,
        random_process=random_process
    )
    agent.compile()

    agent.train(
        env=env,
        episodes=10,
        render=True,
        verbosity=2,
        nb_max_episode_steps=1000,
        plots=False)
