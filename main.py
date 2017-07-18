""" main function """

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time

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
        initializer = tf.random_uniform_initializer(
            -para.init_weight, para.init_weight
        )
        with tf.variable_scope('model', reuse=None, initializer=initializer):
            model = Seq2Seq(para)

        try:
            os.makedirs('models')
        except os.error:
            pass
        print(para)
        sv = tf.train.Supervisor(logdir='./models')
        with sv.managed_session(config=config_setup()) as sess:
            if para.mode == 'train':
                step_time = 0.0
                for step in range(20000):
                    if sv.should_stop():
                        break
                    start_time = time.time()
                    [loss, predict_count, _] = sess.run([
                        model.loss,
                        model.predict_count,
                        model.update
                    ])

                    loss = loss * para.batch_size
                    perplexity = np.exp(loss / predict_count)

                    step_time += (time.time() - start_time)
                    if step % para.steps_per_stats == 0:
                        print('step: %d, perplexity: %.2f step_time: %.2f' %
                              (step, perplexity, step_time / steps_per_stats))
                        step_time = 0
                    step += 1

            elif para.mode == 'test':
                encoder_inputs, encoder_inputs_len = read_testing_sequences(para)

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
