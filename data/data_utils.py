# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile
import json
import pandas as pd
import string
import numpy as np

from six.moves import urllib

from tensorflow.python.platform import gfile
import tensorflow as tf

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

# URLs for WMT data.
_WMT_ENFR_TRAIN_URL = "http://www.statmt.org/wmt10/training-giga-fren.tar"
_WMT_ENFR_DEV_URL = "http://www.statmt.org/wmt15/dev-v2.tgz"

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        line = tf.compat.as_bytes(line)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      # if len(vocab_list) > max_vocabulary_size:
      # vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """

  if tokenizer:
    words = tokenizer(tf.compat.as_bytes(sentence))
  else:
    words = basic_tokenizer(tf.compat.as_bytes(sentence))
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                            normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

#%% Bigram
def parse_title2song_bigram(data_dir, ratio = 0.995):
  if not gfile.Exists(os.path.join(data_dir, 'train.in')):
    with gfile.GFile(os.path.join(data_dir, 'raw_data_filtered.csv'), mode="rb") as f:
      with gfile.GFile(os.path.join(data_dir, 'train.in'), mode="w") as train_in:
          with gfile.GFile(os.path.join(data_dir, 'train.ou'), mode="w") as train_ou:
            with gfile.GFile(os.path.join(data_dir, 'valid.in'), mode="w") as valid_in:
              with gfile.GFile(os.path.join(data_dir, 'valid.ou'), mode="w") as valid_ou:
                data = pd.read_csv(f)
                for i in range(data.shape[0]):
                  d = data.iloc[i].get_values()
                  title = " ".join(re.sub(r'[\u3000-\u303F]', '', str(d[1]).lower()))
                  for c in string.punctuation:
                    title = title.replace(c, '')
                  title = title.split()
                  title_bigram = []
                  for l in range(len(title)-1):
                    title_bigram.append(title[l] + title[l+1])
                  title_bigram = " ".join(title_bigram)
                  songs = re.sub(',', ' ', str(d[2]))
                  train_in.write(title_bigram + '\n')
                  train_ou.write(songs + '\n')
                  if not np.random.rand(1)[0] <= ratio:
                    valid_in.write(title_bigram + '\n')
                    valid_ou.write(songs + '\n')

#%% Unigram
def parse_title2song(data_dir, ratio = 0.995):
  if not gfile.Exists(os.path.join(data_dir, 'train.in')):
    with gfile.GFile(os.path.join(data_dir, 'raw_data_filtered.csv'), mode="rb") as f:
      with gfile.GFile(os.path.join(data_dir, 'train.in'), mode="w") as train_in:
          with gfile.GFile(os.path.join(data_dir, 'train.ou'), mode="w") as train_ou:
            with gfile.GFile(os.path.join(data_dir, 'valid.in'), mode="w") as valid_in:
              with gfile.GFile(os.path.join(data_dir, 'valid.ou'), mode="w") as valid_ou:
                data = pd.read_csv(f)
                for i in range(data.shape[0]):
                  d = data.iloc[i].get_values()
                  title = " ".join(re.sub(r'[\u3000-\u303F]', '', str(d[1]).lower()))
                  for c in string.punctuation:
                    title = title.replace(c, '')
                  songs = re.sub(',', ' ', str(d[2]))
                  train_in.write(title + '\n')
                  train_ou.write(songs + '\n')
                  if not np.random.rand(1)[0] <= ratio:
                    valid_in.write(title + '\n')
                    valid_ou.write(songs + '\n')

#%%
def prepare_playlist_data(data_dir, in_vocabulary_size='_default',
                          ou_vocabulary_size='_default', tokenizer=None):

  parse_title2song_bigram(data_dir)
  ##
  train_path = os.path.join(data_dir, 'train')
  dev_path = os.path.join(data_dir, 'valid')

  # Create vocabularies of the appropriate sizes.
  ou_vocab_path = os.path.join(data_dir, "vocab%s.ou" % ou_vocabulary_size) # output
  in_vocab_path = os.path.join(data_dir, "vocab%s.in" % in_vocabulary_size) # intput
  create_vocabulary(ou_vocab_path, train_path + ".ou", ou_vocabulary_size, tokenizer, False)
  create_vocabulary(in_vocab_path, train_path + ".in", in_vocabulary_size, tokenizer, False)

  # Create token ids for the training data.
  ou_train_ids_path = train_path + (".ids%s.ou" % ou_vocabulary_size)
  in_train_ids_path = train_path + (".ids%s.in" % in_vocabulary_size)
  data_to_token_ids(train_path + ".ou", ou_train_ids_path, ou_vocab_path, tokenizer, False)
  data_to_token_ids(train_path + ".in", in_train_ids_path, in_vocab_path, tokenizer, False)

  # Create token ids for the development data.
  # ou_dev_ids_path = dev_path + (".ids%d.ou" % ou_vocabulary_size)
  # in_dev_ids_path = dev_path + (".ids%d.in" % in_vocabulary_size)
  # data_to_token_ids(dev_path + ".ou", ou_dev_ids_path, ou_vocab_path, tokenizer, False)
  # data_to_token_ids(dev_path + ".in", in_dev_ids_path, in_vocab_path, tokenizer, False)

  return (in_train_ids_path, ou_train_ids_path,
          in_vocab_path, ou_vocab_path)

if __name__ == "__main__":
    prepare_playlist_data('./')
