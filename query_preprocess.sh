# !/bin/bash
echo 'finding OOV words'
python3 find_OOV.py
cd fastText
echo 'fastText'
./fasttext print-word-vectors ./model.bin < ../data/vocab_default.in > ./embedding.txt
./fasttext print-word-vectors ./model.bin < ./queries.txt > ./OOV_embedding.txt
cd ..
echo 'finding alternative words'
python3 gen_alternative_words.py
