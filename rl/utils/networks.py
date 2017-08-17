from keras.layers import Dense, Activation
from keras.models import Model
from keras.layers.merge import concatenate


def simple_actor(env):
    """Build a simple actor network"""
    x = Dense(env.action_space.dim, activation="tanh")(env.state)
    return Model(inputs=[env.state], outputs=[x])


def simple_critic(env):
    """Build a simple critic network"""
    observation = env.state
    action = env.action
    # Concatenate the inputs for the critic
    inputs = concatenate([observation, action])
    x = Dense(1)(inputs)
    x = Activation('linear')(x)

    # Final model
    return Model(inputs=[observation, action], outputs=[x])
