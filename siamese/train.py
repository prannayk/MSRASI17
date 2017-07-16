from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import math
import time
import os
import sys

initializer_main = tf.random_normal_initializer(stddev=0.02)

class ConvSiamese():
	def __init__(self, embedding_size):
		self.embedding_size = embedding_size

	def convolve(self, input, scope):
		
		