# !/bin/bash
cd data
rm *.in
rm *.ou
rm train.tfrecords
echo 'processing raw data'
python3 filter.py
python3 data_utils.py
echo 'processing TensorFlow Standard Format'
python3 tf_format.py
cd ..
