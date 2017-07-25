# !/bin/bash
python3 find_OOV.py
cd fastText
./fasttext print-word-vectors ./model.bin < ./queries.txt > ./OOV_embedding.txt
cd ..
python3 gen_alternative_words.py
