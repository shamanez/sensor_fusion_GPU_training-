# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A binary to train Inception on the ImageNet data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function




import tensorflow as tf

from inception import inception_train
from inception.imagenet_data import ImagenetData


FLAGS = tf.app.flags.FLAGS

import tensorflow as tf 
import numpy as np

import plot

import time
import math
import os
import sys

layers = tf.contrib.layers 

SEPCTURAL_SAMPLES = 10
FEATURE_DIM = SEPCTURAL_SAMPLES*6*2
CONV_LEN = 3
CONV_LEN_INTE = 3#4
CONV_LEN_LAST = 3#5
CONV_NUM = 64
CONV_MERGE_LEN = 8
CONV_MERGE_LEN2 = 6
CONV_MERGE_LEN3 = 4
CONV_NUM2 = 64
INTER_DIM = 120
OUT_DIM = 6#len(idDict)
WIDE = 20
CONV_KEEP_PROB = 0.8

BATCH_SIZE = 64
TOTAL_ITER_NUM = 1000000000

select = 'a'

metaDict = {'a':[119080, 1193], 'b':[116870, 1413], 'c':[116020, 1477]}
TRAIN_SIZE = metaDict[select][0]
EVAL_DATA_SIZE = metaDict[select][1]
EVAL_ITER_NUM = int(math.ceil(EVAL_DATA_SIZE / BATCH_SIZE))






def read_audio_csv(filename_queue):
	reader = tf.TextLineReader()
	key, value = reader.read(filename_queue)
	defaultVal = [[0.] for idx in range(WIDE*FEATURE_DIM + OUT_DIM)]

	fileData = tf.decode_csv(value, record_defaults=defaultVal)
	features = fileData[:WIDE*FEATURE_DIM]
	features = tf.reshape(features, [WIDE, FEATURE_DIM])
	labels = fileData[WIDE*FEATURE_DIM:]
	return features, labels

def input_pipeline(filenames, batch_size, shuffle_sample=True, num_epochs=None):
	filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle_sample)
	# filename_queue = tf.train.string_input_producer(filenames, num_epochs=TOTAL_ITER_NUM*EVAL_ITER_NUM*10000000, shuffle=shuffle_sample)
	example, label = read_audio_csv(filename_queue)
	min_after_dequeue = 1000#int(0.4*len(csvFileList)) #1000
	capacity = min_after_dequeue + 3 * batch_size
	if shuffle_sample:
		example_batch, label_batch = tf.train.shuffle_batch(
			[example, label], batch_size=batch_size, num_threads=16, capacity=capacity,
			min_after_dequeue=min_after_dequeue)
	else:
		example_batch, label_batch = tf.train.batch(
			[example, label], batch_size=batch_size, num_threads=16)
	return example_batch, label_batch




import tensorflow as tf 
import numpy as np

import plot

import time
import math
import os
import sys

layers = tf.contrib.layers 

SEPCTURAL_SAMPLES = 10
FEATURE_DIM = SEPCTURAL_SAMPLES*6*2
CONV_LEN = 3
CONV_LEN_INTE = 3#4
CONV_LEN_LAST = 3#5
CONV_NUM = 64
CONV_MERGE_LEN = 8
CONV_MERGE_LEN2 = 6
CONV_MERGE_LEN3 = 4
CONV_NUM2 = 64
INTER_DIM = 120
OUT_DIM = 6#len(idDict)
WIDE = 20
CONV_KEEP_PROB = 0.8

BATCH_SIZE = 64
TOTAL_ITER_NUM = 1000000000

select = 'a'

metaDict = {'a':[119080, 1193], 'b':[116870, 1413], 'c':[116020, 1477]}
TRAIN_SIZE = metaDict[select][0]
EVAL_DATA_SIZE = metaDict[select][1]
EVAL_ITER_NUM = int(math.ceil(EVAL_DATA_SIZE / BATCH_SIZE))






def read_audio_csv(filename_queue):
	reader = tf.TextLineReader()
	key, value = reader.read(filename_queue)
	defaultVal = [[0.] for idx in range(WIDE*FEATURE_DIM + OUT_DIM)]

	fileData = tf.decode_csv(value, record_defaults=defaultVal)
	features = fileData[:WIDE*FEATURE_DIM]
	features = tf.reshape(features, [WIDE, FEATURE_DIM])
	labels = fileData[WIDE*FEATURE_DIM:]
	return features, labels

def input_pipeline(filenames, batch_size, shuffle_sample=True, num_epochs=None):
	filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle_sample)
	# filename_queue = tf.train.string_input_producer(filenames, num_epochs=TOTAL_ITER_NUM*EVAL_ITER_NUM*10000000, shuffle=shuffle_sample)
	example, label = read_audio_csv(filename_queue)
	min_after_dequeue = 1000#int(0.4*len(csvFileList)) #1000
	capacity = min_after_dequeue + 3 * batch_size
	if shuffle_sample:
		example_batch, label_batch = tf.train.shuffle_batch(
			[example, label], batch_size=batch_size, num_threads=16, capacity=capacity,
			min_after_dequeue=min_after_dequeue)
	else:
		example_batch, label_batch = tf.train.batch(
			[example, label], batch_size=batch_size, num_threads=16)
	return example_batch, label_batch





def select_csv():
	select='a'
	csvFileList = []
	csvDataFolder1 = os.path.join('sepHARData_'+select, "train")
	orgCsvFileList = os.listdir(csvDataFolder1)
	for csvFile in orgCsvFileList:
		if csvFile.endswith('.csv'):
			csvFileList.append(os.path.join(csvDataFolder1, csvFile))



	csvEvalFileList = []
	csvDataFolder2 = os.path.join('sepHARData_'+select, "eval")
	orgCsvFileList = os.listdir(csvDataFolder2)
	for csvFile in orgCsvFileList:
		if csvFile.endswith('.csv'):
			csvEvalFileList.append(os.path.join(csvDataFolder2, csvFile))
	return csvFileList,csvEvalFileList





from data_feed import *

def main(_):
	csvFileList,csvEvalFileList=select_csv()
	#inception_train.train(csvFileList,csvEvalFileList)
  




if __name__ == '__main__':
  tf.app.run()
