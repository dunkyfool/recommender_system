import tensorflow as tf
from utils.initParams import *
from utils.layers import *
import numpy as np

class US_dnn:
  '''
  '''
  def __init__(self, input):
    pass
    self.input = input
    self.weight, self.bias = {}, {}
    self.weight['en_w1'] = weight([8,6],'en_w1')
    self.weight['en_w2'] = weight([6,4],'en_w2')
    self.weight['en_w3'] = weight([4,2],'en_w3')
    self.weight['en_b1'] = bias([6],'en_b1')
    self.weight['en_b2'] = bias([4],'en_b2')
    self.weight['en_b3'] = bias([2],'en_b3')

    self.weight['de_w1'] = tf.transpose(self.weight['en_w3'])
    self.weight['de_w2'] = tf.transpose(self.weight['en_w2'])
    self.weight['de_w3'] = tf.transpose(self.weight['en_w1'])
    self.weight['de_b1'] = bias([4],'de_b1')
    self.weight['de_b2'] = bias([6],'de_b2')
    self.weight['de_b3'] = bias([8],'de_b3')

    self.weight['qq']=True

  def encoder(en_input):
    pass
    layer1 = dnn(en_input, self.weight['en_w1'],self.weight['en_b1'])
    layer2 = dnn(layer1, self.weight['en_w1'],self.weight['en_b1'])
    layer3 = dnn(layer2, self.weight['en_w1'],self.weight['en_b1'])
    return layer3

  def decoder(de_input):
    pass
    layer1 = dnn(de_input, self.weight['de_w1'],self.weight['de_b1'])
    layer2 = dnn(layer1, self.weight['de_w1'],self.weight['de_b1'])
    layer3 = dnn(layer2, self.weight['de_w1'],self.weight['de_b1'])
    return layer3

  def noise(range=[3,5]):
    '''
    1. jiggle correct answer
    '''
    pass

  def run():
    '''
    1. connect model
    2. init/sess
    '''
    pass

  def selftest():
    '''
    1. share variable(learning, identity)
    2. limited level(pass back to control)
    '''
    pass

if __name__=='__main__':
  data = np.array([1,2,3,4,5,6,7,8,
                   2,2,3,4,5,6,7,8]).reshape(2,8)
  x = US_dnn(data)
  print x.weight['en_w1'], x.weight['en_w1'].get_shape()
