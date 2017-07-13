""" main function """

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

from lib.config import params_setup
from lib.seq2seq_model import Seq2Seq
from lib.utils import read_testing_sequences, word_id_to_song_id

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
        with tf.variable_scope('model', reuse=None):
            model = Seq2Seq(para)

        try:
            os.makedirs('models')
        except os.error:
            pass
        print(para)
        sv = tf.train.Supervisor(logdir='./models')
        with sv.managed_session(config=config_setup()) as sess:
            if para.mode == 'train':
                step = 0
                while not sv.should_stop():
                    [loss, _] = sess.run([model.loss, model.update])

                    if step % 100 == 0:
                        print('step: %d, perplexity: %s' % (step, \
                                                            str(np.exp(loss))))
                    step += 1
            elif para.mode == 'test':
                encoder_inputs, encoder_inputs_len = read_testing_sequences(para)

                debug = sess.run(
                    fetches=[
                        model.encoder_inputs,
                        model.encoder_inputs_len,
                        model.encoder_outputs
                    ],
                    feed_dict={
                        model.encoder_inputs: encoder_inputs,
                        model.encoder_inputs_len: encoder_inputs_len
                    }
                )
                for info in debug:
                    print(info)
                    print(info.shape)
                [predicted_ids] = sess.run(
                    fetches=[
                        model.decoder_predicted_ids,
                    ],
                    feed_dict={
                        model.encoder_inputs: encoder_inputs,
                        model.encoder_inputs_len: encoder_inputs_len
                    }
                )
                print(predicted_ids.shape)
                output_file = open('test/out.txt', 'w')
                output_file.write(word_id_to_song_id(para, predicted_ids))
                output_file.close()
