import US_dnn
import FC
import tensorflow as tf
import numpy as np
import pprint as pp
import time
import os
from utils.initParams import *
from utils.layers import *

class panel():
    def __init__(self,):
        pass

    def network(self,x):
        return FC.forward(US_dnn.forward(x))

    def train_US(self,):
        US_dnn.run()

    def train_FC(self,):
        FC.run()

    def train_network(self,MODE=1):
        '''
        1. MODE(1): train only FC
        2. MODE(2): train all
        '''
        pass


if __name__=='__main__':
    pass
