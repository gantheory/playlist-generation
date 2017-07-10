""" convert input data to Standard Tensorflow Format """

import os
import tensorflow as tf
from tqdm import tqdm

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _list_feature(lst):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=lst))

def convert_to_tf_format():
    encoder_file = open('x.txt', 'r')

    encoder_seqs = []
    encoder_seqs_len = []
    for line in encoder_file:
        seq_ids = line.strip().split(' ')
        seq_ids = [1] + [int(id) for id in seq_ids] + [2]
        encoder_seqs.append(seq_ids)
        encoder_seqs_len.append(len(seq_ids) - 1)
    encoder_file.close()
    mx = max(encoder_seqs_len) + 1
    encoder_seqs = [ seq + [0] * (mx - len(seq)) for seq in encoder_seqs ]
    print('encoder max len: %d' % (len(encoder_seqs[0])))

    decoder_file = open('y.txt', 'r')

    decoder_seqs = []
    decoder_seqs_len = []
    for line in decoder_file:
        seq_ids = line.strip().split(' ')
        seq_ids = [1] + [int(id) for id in seq_ids] + [2]
        decoder_seqs.append(seq_ids)
        decoder_seqs_len.append(len(seq_ids) - 1)
    decoder_file.close()
    mx = max(decoder_seqs_len) + 1
    decoder_seqs = [ seq + [0] * (mx - len(seq)) for seq in decoder_seqs ]
    print('decoder max len: %d' % (len(decoder_seqs[0])))

    writer = tf.python_io.TFRecordWriter('train.tfrecords')
    for i in tqdm(range(len(encoder_seqs))):
        example = tf.train.Example(features=tf.train.Features(feature={
            'encoder_input': _list_feature(encoder_seqs[i]),
            'encoder_input_len': _int64_feature(encoder_seqs_len[i]),
            'decoder_input': _list_feature(decoder_seqs[i]),
            'decoder_inputs_len': _int64_feature(decoder_seqs_len[i])
        }))
        writer.write(example.SerializeToString())
    writer.close()

convert_to_tf_format()
