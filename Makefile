.PHONY: all debug test clean

all:
	clear
	python3 main.py

debug:
	clear
	python3 main.py --debug 1

test:
	clear
	python3 main.py --debug 1 --mode test

clean:
	rm models/checkpoint
	rm models/*.local
	rm models/*.pbtxt
	rm models/model.ckpt.*
