from utils.initParams import *
from utils.layers import *
import tensorflow as tf
import numpy as np
import pprint as pp
import time

class US_dnn:
  '''
  '''
  def __init__(self, load=False):
    '''
    *1. laod para
    '''
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

  def encoder(self, en_input):
    layer1 = dnn(en_input, self.weight['en_w1'],self.weight['en_b1'])
    layer2 = dnn(layer1,   self.weight['en_w2'],self.weight['en_b2'])
    layer3 = dnn(layer2,   self.weight['en_w3'],self.weight['en_b3'])
    return layer3

  def decoder(self, de_input):
    layer1 = dnn(de_input, self.weight['de_w1'],self.weight['de_b1'])
    layer2 = dnn(layer1,   self.weight['de_w2'],self.weight['de_b2'])
    layer3 = dnn(layer2,   self.weight['de_w3'],self.weight['de_b3'])
    return layer3

  def noise(self, origin_input, degree=1e-4):
    '''
    1. jiggle correct answer
    '''
    jiggle = np.random.random_sample(origin_input.shape)
    mask = np.random.random_sample(origin_input.shape)
    mask = (mask > 0.5) * 2 -1
    jiggle = mask * jiggle * degree
    return jiggle

  def random_mini_batch(self, origin_input, bs):
    _list = np.random.random_sample(bs) * origin_input.shape[0]
    _list = [int(i) for i in _list]
    return origin_input[_list]

  def run(self, data, lr=0.1, ep=10, mini_bs=1, selftest=False): # ongoing
    '''
    1. connect model
    2. init/sess
    *3. valid data
    *4. save parameter(s) w/ best record
    '''
    x = tf.placeholder(tf.float32,[None,8])
    y_hat = tf.placeholder(tf.float32,[None, 8])

    en_op = self.encoder(data)
    de_op = self.decoder(en_op)

    loss = tf.reduce_sum(tf.square(tf.sub(de_op, y_hat)))
    optimizer = tf.train.RMSPropOptimizer(lr,0.9,0.9,1e-5).minimize(loss)

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        start_time = time.time()

        for epoch in range(ep):
            if selftest: #check share variable
                tmp = sess.run(tf.equal(self.weight['en_w1'],tf.transpose(self.weight['de_w3'])))
                print epoch+1,tmp.all()
                print 'en_para'
                print self.weight['en_w1'].eval()
                print 'de_para'
                print self.weight['de_w3'].eval()

            for i in range(data.shape[0]/mini_bs):
                batch_xs = self.random_mini_batch(data,mini_bs)
                _, output, cost = sess.run([optimizer, de_op, loss],
                                            feed_dict={x:batch_xs,
                                                       y_hat:self.noise(batch_xs)})
            print 'Epoch: %04d, Loss: %.9f' %(epoch+1,cost)
            #pp.pprint(batch_xs)
            #pp.pprint(output)
    print 'Total time: ', time.time()-start_time

  def forward(self, data):
    '''
    1. encoder
    '''
    return self.encoder(data)

  def selftest(self):
    '''
    1. share variable(learning, identity)
    -2. limited level(pass back to control)
    '''
    data = np.array([1,2,3,4,5,6,7,8,
                     2,2,3,4,5,6,7,8]).reshape(2,8)
    self.run(data.astype(np.float32),lr=1e-3,ep=20,selftest=False)

if __name__=='__main__':
  x = US_dnn()
  x.selftest()
