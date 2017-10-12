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



import time
import math
import os
import sys

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('FEATURE_DIM',120,"dimension")
tf.app.flags.DEFINE_integer('CONV_NUM',64,"first convolution output filters")
tf.app.flags.DEFINE_integer('CONV_NUM2',64,"second convolution output filters")
tf.app.flags.DEFINE_integer('OUT_DIM',6,"output classes")
tf.app.flags.DEFINE_integer('WIDE',20,"number of time steps")
tf.app.flags.DEFINE_integer('CONV_KEEP_PROB',0.8,"drop out prob")



#This code is for loading my text files - Data files 
#######################################################################################################################################


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



####################################################################################################################################



def main(_):
	csvFileList,csvEvalFileList=select_csv()
	inception_train.train(csvFileList)
  




if __name__ == '__main__':
  tf.app.run()
