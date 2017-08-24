class Run(object):
    def __init__(self, agent, hooks):
        self.hooks = hooks
        self.agent = agent
        self.run_number = 0

    def __enter__(self):
        self.agent.run_number += 1

    def __exit__(self, *args):
        self.agent.done = True
        self.hooks.run_end()
