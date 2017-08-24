END = "\033[0m"
BG_GREEN = "\033[42m"
FG_RED = "\033[31m"


def print_info(*args):
    print(BG_GREEN, *args, END)


def print_warning(*args):
    print(FG_RED, *args, END)
