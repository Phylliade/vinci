import numpy as np
import matplotlib.pyplot as plt


def portrait_actor(actor, definition=50, plot=True, save_figure=False, figure_file="actor.png"):
    portrait = np.zeros((definition, definition))
    # FIXME: not necessary with env v1?
    center = -0.523
    pos_min = -1.2 - center
    pos_max = 0.6 - center
    max_speed = 0.07
    x_axis = np.linspace(pos_min, pos_max, num=definition)

    for index_x, x in enumerate(x_axis):
        for index_v, v in enumerate(np.linspace(-max_speed, max_speed, num=definition)):
            portrait[index_v, index_x] = actor.predict(np.array([[x, v]]))
    if plot or save_figure:
        plt.figure(figsize=(10, 10))
        plt.imshow(portrait, cmap="inferno", extent=[pos_min, pos_max, -max_speed, max_speed], aspect='auto')
        plt.colorbar()
        plt.scatter([0], [0])
        plt.xlabel("Position")
        plt.ylabel("Velocity")
        if save_figure:
            # TODO: Create the directory if it doesn't exist
            plt.savefig(figure_file)
            plt.close()


def portrait_critic(critic, definition=50, plot=True, action=[-1], save_file="critic.png", save_figure=False, figure_file="actor.png"):
    portrait = np.zeros((definition, definition))
    center = -0.523
    pos_min = -1.2 - center
    pos_max = 0.6 - center
    max_speed = 0.07
    x_axis = np.linspace(pos_min, pos_max, num=definition)

    for index_x, x in enumerate(x_axis):
        for index_v, v in enumerate(np.linspace(-max_speed, max_speed, num=definition)):
            portrait[index_v, index_x] = critic.predict_on_batch([np.array([[x, v]]), np.array(action)])
    if plot or save_figure:
        plt.figure(figsize=(10, 10))
        plt.imshow(portrait, cmap="inferno", extent=[pos_min, pos_max, -max_speed, max_speed], aspect='auto')
        plt.colorbar()
        plt.scatter([0], [0])
        plt.xlabel("Position")
        plt.ylabel("Velocity")
        if save_figure:
            # TODO: Create the directory if it doesn't exist
            plt.savefig(figure_file)
            plt.close()
