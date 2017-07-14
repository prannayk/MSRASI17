from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import math
import time
import os
import sys

graph = tf.Graph()
with graph.as_default():

	train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
