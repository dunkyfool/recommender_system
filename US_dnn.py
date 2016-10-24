from utils.initParams import *
from utils.layers import *
import tensorflow as tf
import numpy as np
import pprint as pp
import time
import os

class US_dnn:
  '''
  '''
  def __init__(self, ):
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
    return origin_input-jiggle

  def random_mini_batch(self, origin_input, bs):
    _list = np.random.random_sample(bs) * origin_input.shape[0]
    _list = [int(i) for i in _list]
    return origin_input[_list]

  def _show_para(self,nameList):
    for name in nameList:
      print name
      print self.weight[name].eval()

  def run(self, data, valid, lr=0.1, ep=10, mini_bs=1, load=False, selftest=False): # ongoing
    '''
    1. connect model
    2. init/sess
    3. valid data
    *4. save 'specific' parameter(s) w/ best record
    +5. load parameter(s)
    '''
    best_record = 9999
    valid_step = 1
    x = tf.placeholder(tf.float32,[None,8])
    y_hat = tf.placeholder(tf.float32,[None, 8])

    en_op = self.encoder(x)
    de_op = self.decoder(en_op)

    loss = tf.reduce_sum(tf.square(tf.sub(de_op, y_hat)))
    optimizer = tf.train.RMSPropOptimizer(lr,0.9,0.9,1e-5).minimize(loss)

    self.saver = tf.train.Saver()

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        if load:
            if not os.path.isfile('tmp/US_dnn.ckpt'):
                print 'There is no saved parameter(s)! The program will assign random weight!'
            else:
                self.saver.restore(sess, "tmp/US_dnn.ckpt")
                best_record = sess.run(loss,feed_dict={x:valid,y_hat:valid})
                print("############# Model restored. #############")
                print 'Current BEST Record', best_record

        start_time = time.time()

        for epoch in range(ep):
            if selftest:
                print '...Before training...'

                ##check share variable
                tmp = sess.run(tf.equal(self.weight['en_w3'],tf.transpose(self.weight['de_w1'])))
                print epoch+1,tmp.all()
                tmp = sess.run(tf.equal(self.weight['en_w2'],tf.transpose(self.weight['de_w2'])))
                print epoch+1,tmp.all()
                self._show_para(['en_w3','de_w1','en_w2','de_w2'])

                ##check load para is correct
                valid_op, valid_cost = sess.run([de_op,loss],feed_dict={x:valid,y_hat:valid})
                print 'Valid_loss: %.9f' %(valid_cost)
                pp.pprint(valid_op)

            for i in range(data.shape[0]/mini_bs):
                batch_xs = self.random_mini_batch(data,mini_bs)
                _, output, cost = sess.run([optimizer, de_op, loss],
                                            feed_dict={x:batch_xs,
                                                       y_hat:self.noise(batch_xs)})

            if (epoch+1) % valid_step == 0:
                valid_op, valid_cost = sess.run([de_op,loss],feed_dict={x:valid,y_hat:valid})
                print 'Epoch: %04d, Loss: %.9f, Valid_loss: %.9f' %(epoch+1,cost,valid_cost)

                if selftest: print '...After training...'; pp.pprint(valid_op);raw_input('Pause')

                if valid_cost < best_record * 0.8:
                    best_record = valid_cost
                    save_path = self.saver.save(sess, "tmp/US_dnn.ckpt")
                    print("############ Model saved in file: %s ###########" % save_path)

                    if selftest: pass; #self._show_para(['en_w3','de_w1'])

    print 'Total time: ', time.time()-start_time

  def forward(self, data):
    '''
    1. encoder
    '''
    return self.encoder(data)

  def selftest(self,load=False,checkpoint=False):
    '''
    1. share variable(learning, identity)
    -2. limited level(pass back to control)
    '''
    data = np.array([1,2,3,4,5,6,7,8,
                     2,2,3,4,5,6,7,8]).reshape(2,8)
    valid =np.array([3,2,3,4,5,6,7,8]).reshape(1,8)
    self.run(data.astype(np.float32),
             valid.astype(np.float32),
             lr=1e-2,ep=20,
             load=load,
             selftest=checkpoint)

if __name__=='__main__':
  x = US_dnn()
  #x.selftest(load=True,checkpoint=True)
  x.selftest(load=True,checkpoint=False)
  #x.selftest(load=False,checkpoint=False)
  #x.selftest(load=False,checkpoint=True)
