""" data processing functions """
import numpy as np
from copy import deepcopy
from collections import defaultdict

__all__ = ['word_id_to_song_id', 'read_testing_sequences', 'read_num_of_seqs']

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

def read_num_of_seqs():
    seqs = open('test/in.txt', 'r').read().splitlines()
    return len(seqs)

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

def check_valid_song_id(song_id):
    filter_list = [ 0, 1, 2, 3, -1]
    return not song_id in filter_list

def word_id_to_song_id(para, predicted_ids):
    dic = open(decoder_vocab_path, 'r').read().splitlines()
    # predicted_ids: [batch_size, <= max_len, beam_width]
    predicted_ids = numpy_array_to_list(predicted_ids)

    # song_id_seqs: [num_of_data * beam_width, <= max_len]
    song_id_seqs = []
    for seq in predicted_ids:
        for i in range(para.beam_width):
            song_id_seqs.append([seq[j][i] for j in range(len(seq))])
    song_id_seqs = [
        [dic[song_id] for song_id in seq if check_valid_song_id(song_id)]
        for seq in song_id_seqs
    ]
    song_id_seqs = [list(set(seq)) for seq in song_id_seqs]

    # merge al of beams
    tmp_seqs = deepcopy(song_id_seqs)
    song_id_seqs = []
    for i in range(int(len(tmp_seqs) / para.beam_width)):
        now = []
        for j in range(para.beam_width):
            now.extend(tmp_seqs[i * para.beam_width + j])
        song_id_seqs.append(list(set(now)))

    return '\n'.join([' '.join(seq) for seq in song_id_seqs])
