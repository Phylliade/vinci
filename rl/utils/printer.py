END = "\033[0m"
BG_GREEN = "\033[42m"
FG_RED = "\033[31m"


def args(f):
    def f_args(*args):
        string = ""
        for arg in args:
            string += str(arg)
        f(string)
    return(f_args)


def print_info(string):
    print(BG_GREEN + string + END)


def print_warning(string):
    print(FG_RED + string + END)
