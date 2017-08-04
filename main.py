""" main function """
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time

import tensorflow as tf
import numpy as np

from lib.config import params_setup
from lib.seq2seq_model import Seq2Seq
from lib.utils import read_testing_sequences, word_id_to_song_id, cal_scores

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
    if para.mode == 'test':
        para.batch_size = 1
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(
            -para.init_weight, para.init_weight
        )
        with tf.variable_scope('model', reuse=None, initializer=initializer):
            model = Seq2Seq(para)

        try:
            os.makedirs(para.model_dir)
        except os.error:
            pass
        print(para)
        sv = tf.train.Supervisor(logdir='./' + para.model_dir)
        with sv.managed_session(config=config_setup()) as sess:
            para_file = open('%s/para.txt' % para.model_dir, 'w')
            para_file.write(str(para))
            para_file.close()
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
                              (step, perplexity, step_time / para.steps_per_stats))
                        step_time = 0
                    step += 1

            elif para.mode == 'test':
                encoder_inputs, encoder_inputs_len = read_testing_sequences(para)

                output_file = open('test/out.txt', 'w')
                scores_file = open('test/scores.txt', 'w')

                total_num = encoder_inputs.shape[0]
                for i in range(total_num):
                    [predicted_ids, decoder_outputs] = sess.run(
                        fetches=[
                            model.decoder_predicted_ids,
                            model.decoder_outputs,
                        ],
                        feed_dict={
                            model.encoder_inputs: encoder_inputs[i:i + 1],
                            model.encoder_inputs_len: encoder_inputs_len[i:i + 1]
                        }
                    )
                    scores = cal_scores(
                        para,
                        predicted_ids,
                        decoder_outputs.beam_search_decoder_output.scores
                    )
                    # print(predicted_ids.shape)
                    # print(scores)
                    output_file.write(word_id_to_song_id(para, predicted_ids) + '\n')
                    scores_file.write('\n'.join(scores) + '\n')

                output_file.close()
                scores_file.close()
