""" main function """

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

from lib.config import params_setup
from lib.seq2seq_model import Seq2Seq

def config_setup():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    return config

if __name__ == "__main__":
    para = params_setup()

    if para.debug == 1:
        para.num_units = 2
        para.num_layers = 2
        para.batch_size = 2
        para.embedding_size = 2
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-para.init_scale,
                                                    para.init_scale)

        with tf.variable_scope('model', reuse=None, initializer=initializer):
            train_model = Seq2Seq(para)

        sv = tf.train.Supervisor(logdir='./models')
        with sv.managed_session(config=config_setup()) as sess:
            if para.mode == 'train':
                epoch = 0
                while not sv.should_stop():
                    [loss, _] = sess.run([train_model.loss, \
                                        train_model.update])

                    if epoch % 100 = 0:
                        print('epoch: %d, loss: %s' % (epoch, str(loss)))
                    epoch += 1
