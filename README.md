# playlist-generation

Implementation of the core( Seq2Seq ) of a palylist generation system.

## Dependencies

* python3
* TensorFlow >= 1.2
* fastText
* numpy
* pandas
* tqdm

## Usage

### Data format reference

* training data: data/raw_data.csv
* testing data: test/in.txt
* training data for fastText: fastText/data.txt

```
$ pip3 install -r requirements.txt
$ # to install TensorFlow, you can refer to https://www.tensorflow.org/install/
```

### Prepare data
```
$ # need to be executed one time before training
$ ./prepare_data.sh
```
### Training
```
$ python3 main.py --mode train # or make (use default parameters)
```

### Pre-trained embedding model
```
$ # To use testing mode of this model, you have to generate an embedding model for OOV words
$ # put your corpus file (data.txt) to fastText/
$ cd fastText
$ make
$ ./fasttext skipgram -input data.txt -output model
```
### Query preprocess
```
$ # need to be excuted one time before testing
$ ./query_preprocess.sh
```
### Testing
```
$ # Usually, we don't use dropout in testing
$ python3 main.py --mode test --dropout 0.0 --model_dir <path_to_your_model>
```

### Other Arguments
If you would like some different settings for this model, you can refer to lib/config.py.
