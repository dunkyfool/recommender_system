from US_dnn import US_dnn
from FC import FC
import tensorflow as tf
import numpy as np
import pprint as pp
import time
import os
from utils.initParams import *
from utils.layers import *

class panel():
    def __init__(self,):
        self.US = US_dnn()
        self.FC = FC()
        pass

    def train_US(self,):
        #US_dnn.run()
        self.US.selftest(load=True,checkpoint=False)

    def train_FC(self,):
        #FC.run()
        self.FC.selftest(load=True,checkpoint=False)

    def train_net(self,MODE=1):
        '''
        1. MODE(1): train only FC
        2. MODE(2): train all
        '''
        US_x = tf.placeholder(tf.float32,[None,8])
        FC_x = tf.placeholder(tf.float32,[None,2])
        US_y_hat = tf.placeholder(tf.float32,[None, 8])
        FC_y_hat = tf.placeholder(tf.float32,[None, 2])

        y1 = self.US.encoder(US_x) # only encoder
        y2 = self.FC.forward(y1)   # for training US-FC

        y3 = self.US.decoder(y1)   # verify US
        y4 = self.FC.forward(FC_x) # verify FC

        loss_US = tf.reduce_sum(tf.square(tf.sub(y3, US_y_hat)))
        loss_FC = tf.reduce_sum(tf.square(tf.sub(y4, FC_y_hat)))

        self.saver_US = tf.train.Saver()
        self.saver_FC = tf.train.Saver()

        with tf.Session() as sess:
            init = tf.initialize_all_variables()
            sess.run(init)

            self.US.loadpara(sess,self.saver_US)
            self.FC.loadpara(sess,self.saver_FC)

            valid_US = np.array([3,2,3,4,5,6,7,8]).reshape(1,8)
            valid_FC = np.array([3,2]).reshape(1,2)
            valid_FC_label = np.array([0,1]).reshape(1,2)

            print sess.run(loss_US,feed_dict={US_x:valid_US,
                                         US_y_hat:valid_US})
            print sess.run(loss_FC,feed_dict={FC_x:valid_FC,
                                         FC_y_hat:valid_FC_label})



if __name__=='__main__':
    x = panel()
    #x.train_US()
    #x.train_FC()
    x.train_net()
