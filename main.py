""" main function """

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

from lib.config import params_setup
from lib.seq2seq import Seq2Seq

def config_setup():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    return config

if __name__ == "__main__":
    para = params_setup()

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-para.init_scale,
                                                    para.init_scale)

        with tf.variable_scope('model', reuse=None, initializer=initializer):
            train_model = Seq2Seq(para)

        sv = tf.train.Supervisor(logdir='/models')
        with sv.managed_session(config=config_setup()) as sess:
            if para.mode == 'train':
                while not sv.should_stop():
                    for i in range(10):
                        print('%d-th iterations' % i)
                        fetches = {}
                        fetches['loss'] = train_model.loss
                        fetches['update'] = train_model.update
                        result = sess.run(fetches)

                        loss = result['loss']
                        print('loss: %s, perplexity %s' % \
                              (str(loss, str(np.exp(loss)))))
