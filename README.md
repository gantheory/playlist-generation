# playlist-generation

Implementation of the core( Seq2Seq ) of a palylist generation system.

## Dependencies

* TensorFlow >= 1.2
* fastText

## Usage

### Data preprocess
```
$ ./data_preprocess.sh
```
### Train
```
$ python3 main.py --mode train
```

### Test
```
$ # Usually, we don't use dropout in testing
$ python3 main.py --mode test --dropout 0.0 --model_dir <path_to_your_model>
```

### Other Arguments
If you would like some different settings for this model, you can refer to lib/config.py.

## Extra things you need

### Data preprocess

To avoid OOV words that may have some influences to the model, we can find similar words in our dictionary from another corpus.

* ./fastText: The directory should contain a pre-trained word embedding model ( you can use your own data set ).
* ./data/embedding.txt: This file should contain word embeddings of words in our dictionary.
