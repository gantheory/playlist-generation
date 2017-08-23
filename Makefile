.PHONY: all debug test clean

train:
	python3 main.py

test:
	python3 main.py --mode test --dropout 0.0

debug:
	python3 main.py --debug 1

debug_test:
	python3 main.py --debug 1 --mode test --dropout 0.0

clean:
	rm data/*.in data/*.ou data/*.txt data/train.tfrecords data/*filtered* test/*filtered* fastText/embedding.txt fastText/OOV_embedding.txt fastText/model.* fastText/queries.txt
	rm -r models
