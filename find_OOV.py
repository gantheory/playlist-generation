""" find OOV words and send a query to fasttext """

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

if __name__ == "__main__":
    testing_file = open('test/in.txt', 'r').read().splitlines()
    seqs = [str_to_bigram_list(seq) for seq in testing_file]

    # a dict check whether a word is in the dictionary or not
    dic = read_dictionary('encoder')
    queries_file = open('fastText/queries.txt', 'w')
    for seq in seqs:
        for word in seq:
            if dic[word] == 3:
                queries_file.write(word + '\n')
    queries_file.close()
