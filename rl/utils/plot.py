import numpy as np
import matplotlib.pyplot as plt


def portrait_actor(actor, definition=100, plot=True):
    portrait = np.zeros((definition, definition))
    for index_x, x in enumerate(np.linspace(-1, 0.6, num=definition)):
        for index_v, v in enumerate(np.linspace(-1, 1, num=definition)):
            portrait[index_x, index_v] = actor.predict([x, v])
    if plot:
        plt.figure(figsize=(10, 10))
        plt.imshow(portrait, cmap="inferno")
        plt.colorbar()
        plt.scatter([definition / 2], [definition / 2])
        plt.xlabel("Position")
        plt.ylabel("Velocity")
