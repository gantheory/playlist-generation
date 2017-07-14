""" data processing functions """
import numpy as np
from collections import defaultdict

encoder_vocab_path = 'data/vocab30000.in'
decoder_vocab_path = 'data/vocab86000.ou'

def str_to_bigram_list(seq):
    return [seq[i] + seq[i + 1] for i in range(len(seq) - 1)]

def read_dictionary(mode):
    if mode == 'encoder':
        file_path = encoder_vocab_path
    else:
        file_path = decoder_vocab_path
    dict_file = open(file_path, 'r').read().splitlines()
    dict_file = [(word, i) for i, word in enumerate(dict_file)]
    dic = defaultdict(lambda: 3)
    for word, idx in dict_file:
        dic[word] = idx
    return dic

def numpy_array_to_list(array):
    if isinstance(array, np.ndarray):
        return numpy_array_to_list(array.tolist())
    elif isinstance(array, list):
        return [numpy_array_to_list(element) for element in array]
    else:
        return array

def read_testing_sequences(para):
    seqs = open('test/in.txt', 'r').read().splitlines()
    seqs = [str_to_bigram_list(seq) for seq in seqs]

    dic = read_dictionary('encoder')
    seqs = [[dic[word] for word in seq] for seq in seqs]
    seqs = [seq + [2] for seq in seqs]

    seqs_len = [len(seq) for seq in seqs]
    seqs = [np.array(seq + [0] * (para.max_len - len(seq))) for seq in seqs]
    para.batch_size = len(seqs)
    print('total num of sequences: %d' % len(seqs))

    return np.asarray(seqs), np.asarray(seqs_len)

def word_id_to_song_id(para, predicted_ids):
    dic = open(decoder_vocab_path, 'r').read().splitlines()
    # predicted_ids: [batch_size, <= max_len, beam_width]
    predicted_ids = numpy_array_to_list(predicted_ids)

    song_ids_str = ''
    for seq in predicted_ids:
        now_song_ids = [''] * para.beam_width
        for i, beam_ids in enumerate(seq):
            now_song_ids = [now_song_ids[j] + str(beam_ids[j]) \
                            for j in range(len(beam_ids))]
            if i != len(seq) - 1:
                now_song_ids = [ids + ' ' for ids in now_song_ids]
            else:
                now_song_ids = [ids + '\n' for ids in now_song_ids]
        song_ids_str += ''.join(now_song_ids)

    return song_ids_str
