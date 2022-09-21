# main.py
#
# Read in training data, initialize GAN, train

import sys
import cv2
import os
import numpy as np
import configparser
from model import ELFVS_model
import tensorflow as tf
import pandas as pd

# avoid warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# avoid error messages
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess =tf.compat.v1.InteractiveSession(config=config)
physical_devices = tf.config.experimental.list_physical_devices('GPU')

# # read in config
# config = configparser.ConfigParser()
# config.read('config.ini')
# data_dir = config['Data']['data_dir']
# training_videos = eval(config['Data']['training_videos'])
# testing_videos = eval(config['Data']['testing_videos'])
# num_rows = int(config['LF_image']['num_rows'])
# num_cols = int(config['LF_image']['num_cols'])
# num_channels = int(config['LF_image']['num_channels'])
# height = int(config['LF_image']['height'])
# width = int(config['LF_image']['width'])
# epochs = int(config['Training']['epochs'])
# batch_size = int(config['Training']['batch_size'])

if __name__ == '__main__':


	if len(sys.argv) == 2 and sys.argv[1] == "test":
		ELFVS = ELFVS_model(session=sess, training=False)
		ELFVS.test()

	else:
		ELFVS = ELFVS_model(session=sess)
		ELFVS.train()
