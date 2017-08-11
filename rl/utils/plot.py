import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from .stats import network_values

# TODO: Use an unified sample set for portrait_actor, portrait_critic and plot_distribution
# e.g. a meshgrid


def portrait_actor(actor, env, figure=None, definition=50, plot=True, save_figure=False, figure_file="actor.png"):
    """Portrait the actor"""
    if env.observation_space.dim != 2:
        raise(ValueError("The provided environment has an observation space of dimension {}, whereas it should be 2".format(env.observation_space.dim)))

    portrait = np.zeros((definition, definition))
    x_min, y_min = env.observation_space.low
    x_max, y_max = env.observation_space.high
    # Use the dimension names if given otherwise default to "x" and "y"
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])

    for index_x, x in enumerate(np.linspace(x_min, x_max, num=definition)):
        for index_y, y in enumerate(np.linspace(y_min, y_max, num=definition)):
            # Be careful to fill the matrix in the right order
            portrait[definition - (1 + index_y), index_x] = actor.predict(np.array([[x, y]]))
    if plot or save_figure:
        if figure is None:
            plt.figure(figsize=(10, 10))
        plt.imshow(portrait, cmap="inferno", extent=[x_min, x_max, y_min, y_max], aspect='auto')
        plt.colorbar(label="action")
        # Add a point at the center
        plt.scatter([0], [0])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title("Actor phase portrait")
        if save_figure:
            # TODO: Create the directory if it doesn't exist
            plt.savefig(figure_file)
            plt.close()


def portrait_critic(critic, env, figure=None, definition=50, plot=True, action=[-1], save_figure=False, figure_file="critic.png"):
    if env.observation_space.dim != 2:
        raise(ValueError("The provided environment has an observation space of dimension {}, whereas it should be 2".format(env.observation_space.dim)))

    portrait = np.zeros((definition, definition))
    x_min, y_min = env.observation_space.low
    x_max, y_max = env.observation_space.high
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])

    for index_x, x in enumerate(np.linspace(x_min, x_max, num=definition)):
        for index_y, y in enumerate(np.linspace(y_min, y_max, num=definition)):
            # Be careful to fill the matrix in the right order
            portrait[definition - (1 + index_y), index_x] = critic.predict_on_batch([np.array([[x, y]]), np.array(action)])
    if plot or save_figure:
        if figure is None:
            figure = plt.figure(figsize=(10, 10))
        plt.imshow(portrait, cmap="inferno", extent=[x_min, x_max, y_min, y_max], aspect='auto')
        plt.colorbar(label="critic value")
        # Add a point at the center
        plt.scatter([0], [0])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title("Critic phase portrait")
        if save_figure:
            # TODO: Create the directory if it doesn't exist
            plt.savefig(figure_file)
            plt.close()


def plot_trajectory(trajectory, actor, env, figure=None, figure_file="trajectory.png", definition=50, plot=True, save_figure=False,):
    if figure is None:
        plt.figure(figsize=(10, 10))
    plt.scatter(trajectory["x"], trajectory["y"], c=range(1, len(trajectory["x"]) + 1))
    plt.colorbar(orientation="horizontal", label="steps")

    if env.observation_space.dim != 2:
        raise(ValueError("The provided environment has an observation space of dimension {}, whereas it should be 2".format(env.observation_space.dim)))

    # Add the actor phase portrait
    portrait = np.zeros((definition, definition))
    x_min, y_min = env.observation_space.low
    x_max, y_max = env.observation_space.high
    # Use the dimension names if given otherwise default to "x" and "y"
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])

    for index_x, x in enumerate(np.linspace(x_min, x_max, num=definition)):
        for index_y, y in enumerate(np.linspace(y_min, y_max, num=definition)):
            # Be careful to fill the matrix in the right order
            portrait[definition - (1 + index_y), index_x] = actor.predict(np.array([[x, y]]))

    plt.imshow(portrait, cmap="inferno", extent=[x_min, x_max, y_min, y_max], aspect='auto')
    plt.colorbar(label="action")
    # Add a point at the center
    plt.scatter([0], [0])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(figure_file)
    plt.close()


def plot_distribution(actor, critic, env, actor_file="actor_distribution.png", critic_file="critic_distribution.png"):
    """Plot the distributions of the network values"""
    actor_actions, critic_values = network_values(env, actor, critic)

    plot_action_distribution(actor_actions, actor_file)
    plot_value_distribution(critic_values, critic_file)


def plot_action_distribution(actions, file="action_ditribution.png"):
    plt.figure(figsize=(10, 10))
    sb.distplot(actions, kde=False)
    plt.ylabel("probability")
    plt.xlabel("action")
    plt.title("Action distribution")
    plt.savefig(file)
    plt.close()


def plot_value_distribution(values, file="value_distribution.png"):
    plt.figure(figsize=(10, 10))
    sb.distplot(values)
    plt.xlabel("critic value")
    plt.title("Value distribution")
    plt.savefig(file)
    plt.close()
