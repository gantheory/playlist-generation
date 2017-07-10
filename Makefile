all:
	python3 main.py

clean:
	rm models/checkpoint
	rm models/*.local
	rm models/*.pbtxt
	rm models/model.ckpt.*
