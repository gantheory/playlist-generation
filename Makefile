.PHONY: all debug test clean

all:
	clear
	python3 main.py

test:
	clear
	python3 main.py --mode test --dropout 0.0

debug:
	clear
	python3 main.py --debug 1 --model_dir models

debug_test:
	clear
	python3 main.py --debug 1 --mode test --dropout 0.0 --model_dir models

clean:
	rm models/checkpoint
	rm models/*.local
	rm models/*.pbtxt
	rm models/model.ckpt.*
