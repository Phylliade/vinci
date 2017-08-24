class Run():
    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.done = True
        self.hooks.run_end()
