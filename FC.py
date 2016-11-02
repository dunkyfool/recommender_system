from utils.initParams import *
from utils.layers import *
import tensorflow as tf
import numpy as np
import pprint as pp
import time
import os

class FC():
  '''
  '''
  def __init__(self, ):
    self.weight, self.bias = {}, {}
    with tf.name_scope('weights'):
        # declare variable
        self.weight['fc_w1'] = weight([2,4],'fc_w1')
        self.weight['fc_w2'] = weight([4,2],'fc_w2')
        self.weight['fc_b1'] = bias([4],'fc_b1')
        self.weight['fc_b2'] = bias([2],'fc_b2')

        # para list for saver
        self.paraList = [self.weight['fc_w1'],
                         self.weight['fc_w2'],
                         self.weight['fc_b1'],
                         self.weight['fc_b2']]

        # declare tensorboard variable
        self.variable_summaries(self.weight['fc_w1'], 'fc_w1')
        self.variable_summaries(self.weight['fc_w2'], 'fc_w2')
        self.variable_summaries(self.weight['fc_b1'], 'fc_b1')
        self.variable_summaries(self.weight['fc_b2'], 'fc_b2')

  def encoder(self, fc_input):
    layer1 = dnn(fc_input, self.weight['fc_w1'],self.weight['fc_b1'])
    layer2 = dnn(layer1,   self.weight['fc_w2'],self.weight['fc_b2'])
    return layer2

  def random_mini_batch(self, origin_input, origin_label, bs):
    _list = np.random.random_sample(bs) * origin_input.shape[0]
    _list = [int(i) for i in _list]
    return origin_input[_list], origin_label[_list]

  def _show_para(self,nameList):
    for name in nameList:
      print name
      print self.weight[name].eval()

  def variable_summaries(self,var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.scalar_summary('stddev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)

  def loadpara(self,sess,saver):
    saver.restore(sess, "tmp/FC.ckpt")

  def run(self, data, data_label, valid, valid_label, lr=0.1, ep=10, mini_bs=1, load=False, selftest=False): # ongoing
    '''
    1. connect model
    2. init/sess
    3. valid data
    *4. save 'specific' parameter(s) w/ best record
    5. load parameter(s)
    *6. tensorboard
    '''
    best_record = 9999
    valid_step = 1
    logs_path = 'logs/'
    x = tf.placeholder(tf.float32,[None,2])
    y_hat = tf.placeholder(tf.float32,[None, 2])

    with tf.name_scope('Model'):
        de_op = self.encoder(x)

    with tf.name_scope('Loss'):
        loss = tf.reduce_sum(tf.square(tf.sub(de_op, y_hat)))

    with tf.name_scope('RMSProp'):
        optimizer = tf.train.RMSPropOptimizer(lr,0.9,0.9,1e-5).minimize(loss)

    tf.scalar_summary("loss", loss)
    merged_summary_op = tf.merge_all_summaries()

    self.saver = tf.train.Saver(self.paraList)

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
        if load:
            if not os.path.isfile('tmp/FC.ckpt'):
                print 'There is no saved parameter(s)! The program will assign random weight!'
            else:
                self.loadpara(sess,self.saver)
                best_record = sess.run(loss,feed_dict={x:valid,y_hat:valid_label})
                print("############# Model restored. #############")
                print 'Current BEST Record', best_record

        start_time = time.time()

        for epoch in range(ep):
            if selftest:
                print '...Before training...'

                ##check variable
                self._show_para(['fc_w1','fc_w2'])

                ##check load para is correct
                valid_op, valid_cost = sess.run([de_op,loss],feed_dict={x:valid,y_hat:valid_label})
                print 'Valid_loss: %.9f' %(valid_cost)
                pp.pprint(valid_op)

            for i in range(data.shape[0]/mini_bs):
                batch_xs, batch_ys = self.random_mini_batch(data,data_label,mini_bs)
                _, output, cost, summary = sess.run([optimizer, de_op, loss, merged_summary_op],
                                                     feed_dict={x:batch_xs,
                                                                y_hat:batch_ys})
                summary_writer.add_summary(summary,epoch*(data.shape[0]/mini_bs)+i)

            if (epoch+1) % valid_step == 0:
                valid_op, valid_cost = sess.run([de_op,loss],feed_dict={x:valid,y_hat:valid_label})
                print 'Epoch: %04d, Loss: %.9f, Valid_loss: %.9f' %(epoch+1,cost,valid_cost)

                if selftest: print '...After training...'; pp.pprint(valid_op);raw_input('Pause')

                if valid_cost < best_record * 0.8:
                    best_record = valid_cost
                    save_path = self.saver.save(sess, "tmp/FC.ckpt")
                    print("############ Model saved in file: %s ###########" % save_path)

                    if selftest: pass; self._show_para(['fc_w1','fc_w2'])

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
    data = np.array([1,2,
                     2,2,]).reshape(2,2)
    data_label = np.array([1,0,0,1]).reshape(2,2)
    valid =np.array([3,2]).reshape(1,2)
    valid_label = np.array([0,1]).reshape(1,2)

    self.run(data.astype(np.float32),data_label,
             valid.astype(np.float32),valid_label,
             lr=1e-2,ep=20,
             load=load,
             selftest=checkpoint)

if __name__=='__main__':
  x = FC()
  #x.selftest(load=True,checkpoint=True)
  x.selftest(load=True,checkpoint=False)
  #x.selftest(load=False,checkpoint=False)
  #x.selftest(load=False,checkpoint=True)
