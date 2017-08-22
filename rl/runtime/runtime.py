def run(self, epochs):
    self.training = True
    self.done = True

    # We could use a termination criterion, based on step instead of epoch, as in  _run
    # TODO: Trigger an exception
    for epoch in range(epochs):
        if self.done:
            self.episode += 1
            self.episode_reward = 0.
            self.episode_step = 0

        # Initialize the step
        self.done = False
        self.step += 1
        self.episode_step += 1
        self.step_summaries = []

        # Run the step
        yield epoch

        # Close the step


def experiments(epochs):
    for epoch in range(epochs):
        if (epoch + 1) == epochs:
            experiment.done = True
        yield epoch
