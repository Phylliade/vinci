import scipy.stats


def entropy(p):
    return(scipy.stats.entropy(p))


def kl_divergence(p, q):
    return(scipy.stats.entropy(p, q))

def mutual_information(x, y):
    pass
