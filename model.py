import cv2
import numpy as np
import os
import random
import configparser
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, ConvLSTM2D
from tensorflow.keras.layers import Input, Dense, Reshape, Dropout, concatenate, MaxPooling2D, UpSampling2D, BatchNormalization, Conv2DTranspose
from tensorflow.keras.layers import Conv2D, Conv3D, GaussianNoise, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model, load_model
from datetime import datetime


class My_Loss(tf.keras.losses.Loss):
	def __init__(self):
		super().__init__()
	def call(self, y_true, y_pred):
		y_true = tf.reshape(y_true, [203416, 3])
		y_pred = tf.reshape(y_pred, [203416, 3])
		loss = tf.constant(0.0)
		condition = lambda i, loss: i < 203416
		body = lambda i, loss: (tf.add(i,1), tf.add(loss, y_true[i][0] * tf.math.log(y_pred[i][0])))
		i, total_loss = tf.while_loop(condition, body, (tf.constant(0), loss))
		print(total_loss)
		cce = tf.keras.losses.CategoricalCrossentropy()
		return cce(y_true, y_pred)



class ELFVS_model(object):
	"""docstring for LF_edit_model"""
	def __init__(self, session, training=True):
		# read in config
		config = configparser.ConfigParser()
		config.read('config.ini')
		self.data_dir = config['Data']['data_dir']
		self.training_videos = eval(config['Data']['training_videos'])
		self.testing_videos = eval(config['Data']['testing_videos'])
		self.num_rows = int(config['LF_image']['num_rows'])
		self.num_cols = int(config['LF_image']['num_cols'])
		self.num_channels = int(config['LF_image']['num_channels'])
		self.height = int(config['LF_image']['height'])
		self.width = int(config['LF_image']['width'])
		self.epochs = int(config['Training']['epochs'])
		self.batch_size = int(config['Training']['batch_size'])
		self.timesteps = int(config['Training']['timesteps'])
		self.rgb_events_ratio = int(config['Training']['rgb_events_ratio'])
		self.num_bins = int(config['Training']['num_bins'])
		checkpoints = config['Training']['checkpoints']

		# build model for training and testing
		if training:
			if os.path.isfile('checkpoints/model.h5') and checkpoints:
				self.model = load_model('checkpoints/model.h5')
			else:
				self.model = self.build_model()
				self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer='sgd')




	def build_model(self):
		rgb_input_central = Input(shape=(2, self.height, self.width, 3))
		rgb_input_side = Input(shape=(2, self.height, self.width, 3))
		events_input = Input(shape=(self.num_bins, self.height,self.width, 1))
		events_output = Input(shape=(self.height,self.width, 3))

		rgbs = concatenate([rgb_input_central, rgb_input_side], axis=1)
		rgb_features = Conv3D(16, (3,3,3), padding='same')(rgbs)
		rgb_features = Conv3D(32, (4,1,1), padding='valid')(rgbs)
		rgb_features_E1 = Reshape([self.height, self.width, 32])(rgb_features)
		rgb_features_E2 = MaxPooling2D((2,2))(rgb_features_E1)
		rgb_features_E2 = Conv2D(64, (3,3), padding='same')(rgb_features_E2)

		E0_output = ConvLSTM2D(32, (3,3), padding='same')(events_input)

		E1_input = concatenate([E0_output, rgb_features_E1])
		E1_input = Reshape([1, self.height, self.width, 64])(E1_input)
		E1_output = ConvLSTM2D(64, (3,3), padding='same')(E1_input)
		E1_output = MaxPooling2D((2,2))(E1_output)

		E2_input = concatenate([E1_output, rgb_features_E2])
		E2_input = Reshape([1, self.height//2, self.width//2, 128])(E2_input)
		E2_output = ConvLSTM2D(128, (3,3), padding='same')(E2_input)
		E2_output = MaxPooling2D((2,2))(E2_output)

		D2_input = E2_output
		D2_output = Conv2DTranspose(64, (3,3), padding='same')(D2_input)
		D2_output = UpSampling2D((2,2))(D2_output)

		D1_input = concatenate([E1_output, D2_output])
		D1_output = Conv2DTranspose(32, (3,3), padding='same')(D1_input)
		D1_output = UpSampling2D((2,2))(D1_output)
		D1_output = ZeroPadding2D(padding=((0,0), (1,0)))(D1_output)

		D0_input = concatenate([E0_output, D1_output])
		output = Dense(3, activation='softmax')(D0_input)

		model = Model(inputs=[rgb_input_central, rgb_input_side, events_input, events_output], outputs=output)

		model.summary()
		for layer in model.layers:
			print(layer.get_output_at(0).get_shape().as_list())

		return model




	def train(self):
		# create log file
		time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
		log_file = open('checkpoints/logs/%s.txt' % time, 'a+')

		# Training
		for epoch in range(self.epochs):
			# read in data for this epoch
			num_videos = len(self.training_videos)
			# inputs and outputs
			rgb_input_central = np.zeros((self.batch_size, int(self.timesteps/self.rgb_events_ratio), self.height, self.width, 3), dtype=np.uint8)
			rgb_input_side = np.zeros((self.batch_size, int(self.timesteps/self.rgb_events_ratio), self.height, self.width, 3), dtype=np.uint8)
			events_input = np.zeros((self.batch_size, self.timesteps, self.height, self.width, 3), dtype=np.uint8)
			events_output = np.zeros((self.batch_size, self.height, self.width, 3), dtype=np.uint8)


			for batch in range(self.batch_size):
				# random video, frames, row and column
				rand_video = self.training_videos[random.randint(0, num_videos-1)]
				rand_frames = self.get_rand_frames(rand_video)
				rand_row = random.randint(0, 6)
				rand_col = random.randint(0, 6)

				
				for index, rand_frame in enumerate(rand_frames):
					if int(rand_frame)%3 == 0:
						rgb_input_central[batch, int(index/self.rgb_events_ratio)] = cv2.imread(os.path.join(self.data_dir, rand_video, rand_frame, 'original', '3_3.png'))
						# cv2.imwrite('rgb_input_central{}.png'.format(int(index/self.rgb_events_ratio)), rgb_input_central[batch, int(index/self.rgb_events_ratio)])
						rgb_input_side[batch, int(index/self.rgb_events_ratio)] = cv2.imread(os.path.join(self.data_dir, rand_video, rand_frame, 'original', '%d_%d.png'%(rand_row, rand_col)))
						# cv2.imwrite('rgb_input_side{}.png'.format(int(index/self.rgb_events_ratio)), rgb_input_side[batch, int(index/self.rgb_events_ratio)])

					events_input[batch, index] = cv2.imread(os.path.join(self.data_dir, rand_video, rand_frame, 'modified', '3_3.png'))

				events_output[batch] = cv2.imread(os.path.join(self.data_dir, rand_video, rand_frames[-1], 'modified', '%d_%d.png'%(rand_row, rand_col)))

			events_input = self.to_event_voxel_grid(events_input)
			events_output = np.array([[[[1.0,0.0,0.0] if pixel[0]==255 else [0.0,0.0,1.0] if pixel[2]==255 else [0.0,1.0,0.0] for pixel in row] for row in batch_imgs] for batch_imgs in events_output])
			
			for iteration in range(50):
				loss = self.model.train_on_batch([rgb_input_central, rgb_input_side, events_input, events_output], events_output)
				output_str = "Epoch %d iteration %d: %s" % (epoch, iteration, loss)
				print(output_str)
				log_file.write(output_str + '\n')

			self.model.save('checkpoints/model.h5')

		log_file.close()




	def get_rand_frames(self, rand_video):
		all_frames = os.listdir(os.path.join(self.data_dir, rand_video))
		all_frames.sort()

		rand_start = random.randint(0, len(all_frames)-self.timesteps)

		rand_frames = []

		for i in range(self.timesteps):
			rand_frames.append(all_frames[rand_start + i])

		return rand_frames



	# convert input into event voxel grid
	def to_event_voxel_grid(self, events_input):
		delta_t = events_input.shape[1]
		output = np.zeros((self.batch_size, self.num_bins, self.height, self.width, 1))

		for batch in range(self.batch_size):
			for _bin in range(self.num_bins):
				sums = np.array([[events_input[batch,timestamp,:,:,0] * max(0, 1 - abs(_bin - (self.num_bins-1)/delta_t*timestamp)) / 255 - events_input[batch,timestamp,:,:,2] * max(0, 1 - abs(_bin - (self.num_bins-1)/delta_t*timestamp)) / 255 for timestamp in range(delta_t)] for batch in range(len(events_input))]).sum(axis=1)
				sums = sums.reshape((self.height, self.width, 1))
				output[batch][_bin] = sums

		return output



	def test(self):
		self.model = load_model('checkpoints/model.h5')
		time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
		os.mkdir(os.path.join("outputs", time))

		# read in data for this epoch
		num_videos = len(self.testing_videos)
		# inputs and outputs
		rgb_input_central = np.zeros((self.batch_size, int(self.timesteps/self.rgb_events_ratio), self.height, self.width, 3))
		rgb_input_side = np.zeros((self.batch_size, int(self.timesteps/self.rgb_events_ratio), self.height, self.width, 3))
		events_input = np.zeros((self.batch_size, self.timesteps, self.height, self.width, 3))
		events_output = np.zeros((self.batch_size, self.height, self.width, 3))


		for batch in range(self.batch_size):
			# random video, frames, row and column
			rand_video = self.testing_videos[random.randint(0, num_videos-1)]
			rand_frames = self.get_rand_frames(rand_video)
			rand_row = random.randint(0, 6)
			rand_col = random.randint(0, 6)

			
			for index, rand_frame in enumerate(rand_frames):
				if int(rand_frame)%3 == 0:
					rgb_input_central[batch, int(index/self.rgb_events_ratio)] = cv2.imread(os.path.join(self.data_dir, rand_video, rand_frame, 'original', '3_3.png'))
					rgb_input_side[batch, int(index/self.rgb_events_ratio)] = cv2.imread(os.path.join(self.data_dir, rand_video, rand_frame, 'original', '%d_%d.png'%(rand_row, rand_col)))
				events_input[batch, index] = cv2.imread(os.path.join(self.data_dir, rand_video, rand_frame, 'modified', '3_3.png'))
			events_output[batch] = cv2.imread(os.path.join(self.data_dir, rand_video, rand_frames[-1], 'modified', '%d_%d.png'%(rand_row, rand_col)))

		events_input = self.to_event_voxel_grid(events_input)

		results = self.model.predict([rgb_input_central, rgb_input_side, events_input, events_output])
		for idx, img in enumerate(results):
			batch = str(idx)
			os.mkdir(os.path.join("outputs", time, batch))
			img = np.array([[[255,0,0] if np.argmax(pixel)==0 else [0,0,255] if np.argmax(pixel)==2 else [0,0,0] for pixel in row] for row in img])
			cv2.imwrite(os.path.join('outputs', time, batch, 'predicted.png'), img)
			cv2.imwrite(os.path.join('outputs', time, batch, 'ground_truth.png'), events_output[idx])
			print(np.amin(img))
			print(np.amax(img))
			print(np.mean(img))
			print(np.amin(events_output[idx]))
			print(np.amax(events_output[idx]))
			print(np.mean(events_output[idx]))