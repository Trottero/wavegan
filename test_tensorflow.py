# Simple script prints out the available GPUs for tensorflow 

from __future__ import print_function
try:
  import cPickle as pickle
except:
  import pickle
from functools import reduce
import os
import time

import numpy as np
import tensorflow as tf
from six.moves import xrange

import loader
from wavegan import WaveGANGenerator, WaveGANDiscriminator
import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
