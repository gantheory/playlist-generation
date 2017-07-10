""" main function """

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

from lib.config import params_setup
from lib.seq2seq import Seq2Seq

if __name__ == "__main__":
    para = params_setup()

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-0.1, 0.1)

        with tf.variable_scope('model', reuse=None, initializer=initializer):
            train_model = Seq2Seq(para)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        sv = tf.train.Supervisor(logdir='models')
        with sv.managed_session(config=config) as sess:
            for i in range(10):
                print('%d-th iterations' % i)
                [loss, update] = sess.run([train_model.loss, train_model.update])

                print('loss: %s, perplexity %s' % (str(loss, str(np.exp(loss)))))
