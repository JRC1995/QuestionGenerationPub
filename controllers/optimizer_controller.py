from optimizers import *

def get_optimizer(config):
    return eval(config["optimizer"])


