""" find OOV words and send a query to fasttext """

from collections import defaultdict
from subprocess import call

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

def cos_similarity(x, y):
    sx2 = 0.0
    sy2 = 0.0
    sxy = 0.0
    for i in range(len(x)):
        sx2 += x[i] ** 2
        sy2 += y[i] ** 2
        sxy += x[i] * y[i]
    sx2 = sx2 ** 0.5
    sy2 = sy2 ** 0.5
    return sxy / sx2 / sy2

def find_another_word(word, model, original_embedding, id_to_word, output_file):
    now_vector = model[word]

    max_similarity = 0.0
    match_id = -1
    similarity = []

    for i in range(len(original_embedding)):
        similarity.append(cos_similarity(now_vector, original_embedding[i]))
    for i in range(len(similarity)):
        if similarity[i] > mx:
            mx = similarity[i]
            match_id = i
    if mx >= 0.8:
        output_file.wrtie(word + ' ' + id_to_word[match_id] + '\n')

if __name__ == "__main__":
    oov_file = open('fastText/OOV_embedding.txt', 'r').read().splitlines()
    oov_file = [seq.split(' ') for seq in oov_file]
    oov_word = [seq[0] for seq in oov_file]
    oov_embedding = [seq[1:] for seq in oov_file]
    for i in range(len(oov_embedding)):
        for j in range(len(oov_embedding[i])):
            if len(oov_embedding[i][j]) == 0:
                oov_embedding[i][j] = 0.0
            else:
                oov_embedding[i][j] = float(oov_embedding[i][j])
    oov_embedding = [seq[:100] for seq in oov_embedding]

    original_embedding = open('data/embedding.txt', 'r').read().splitlines()
    original_embedding = [seq.split(' ') for seq in original_embedding]
    for i in range(len(original_embedding)):
        for j in range(len(original_embedding[i])):
            original_embedding[i][j] = float(original_embedding[i][j])

    id_to_word = open(encoder_vocab_path, 'r').read().splitlines()
    output_file = open('test/alternative_words.txt', 'w')
    for i in range(len(oov_word)):
        max_similarity = 0.0
        match_id = -1
        for j in range(len(original_embedding)):
            similarity = cos_similarity(oov_embedding[i], original_embedding[j])
            if similarity > max_similarity:
                max_similarity = similarity
                match_id = j
        if max_similarity >= 0.8:
            output_file.write(oov_word[i] + ' ' + id_to_word[match_id] + '\n')
    output_file.close()
