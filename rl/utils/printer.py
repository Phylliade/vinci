from __future__ import print_function

END = "\033[0m"
BG_GREEN = "\033[42m"
FG_RED = "\033[31m"
BG_MAGENTA = "\033[45m"


def args(f):
    def f_args(*args):
        string = ""
        for arg in args:
            string += str(arg)
        f(string)
    return(f_args)


def print_info(string):
    print(BG_GREEN + str(string) + END)


def print_warning(string):
    print(FG_RED + str(string) + END)


def print_debug(string):
    """
    Used to display debug statements.
    These statements are only temporary, to make code work
    """
    print(BG_MAGENTA + str(string) + END)


def print_status(string, terminal=False):
    """Print with no newline"""
    if terminal:
        end = "\n"
    else:
        end = ""
    print("\r" + str(string), end=end)


def print_epoch(epoch, nb_epochs):
    print_status("Iteration {}/{}".format(epoch, nb_epochs), terminal=(epoch == nb_epochs))
