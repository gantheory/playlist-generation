# !/bin/bash
cd data
rm *.in *.ou train.tfrecords *filtered*
echo 'processing raw data'
python3 data_utils.py
echo 'processing TensorFlow Standard Format'
python3 tf_format.py
cd ..
