# !/bin/bash
cd data
echo 'processing raw data'
python3 filter.py
python3 data_utils.py
echo 'processing TensorFlow Standard Format'
python3 tf_format.py
cd ..
echo 'finding OOV words'
python3 find_OOV.py
cd fastText
echo 'fastText'
./fasttext print-word-vectors ./model.bin < ./queries.txt > ./OOV_embedding.txt
cd ..
echo 'finding alternative words'
python3 gen_alternative_words.py
