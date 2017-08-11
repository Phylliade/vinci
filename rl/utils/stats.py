import numpy as np


def network_values(env, actor, critic, definition=10000):
    states = np.random.uniform(low=env.observation_space.low, high=env.observation_space.high, size=((definition, env.observation_space.dim)))
    actions = np.random.uniform(low=env.action_space.low, high=env.action_space.high, size=((definition, env.action_space.dim)))
    distribution_actor = actor.predict_on_batch([states])[:, 0]
    distribution_critic = critic.predict_on_batch([states, actions])

    return((distribution_actor, distribution_critic))
