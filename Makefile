all:
	clear
	python3 main.py

debug:
	clear
	python3 main.py --debug 1

clean:
	rm models/checkpoint
	rm models/*.local
	rm models/*.pbtxt
	rm models/model.ckpt.*
