import math
import numpy as np
import scipy.io as sio
import tensorflow as tf
import config


class DataItem:
	pass

class DataHandler:
	def __init__(self):
		self.cfg=config.cfg
		self.data = self._load_dataset()
		self._extend_cfg()
		

	def _load_dataset(self):
			# Load Matlab file dataset annotation
			mlab = sio.loadmat(self.cfg.matfilepath)
			raw_data = mlab
			# print(raw_data)
			mlab=mlab['data_compiler']
			self.num_images = mlab.shape[1]
			data = []

			sample = mlab[0,1]
			for i in range(self.num_images):
				sample = mlab[0, i]

				item = DataItem()
				item.image_id = i
				item.im_path = sample[0][0]
				item.zdepth = sample[1][0][0]
				data.append(item)


			imgpaths=tf.constant([item.im_path for item in data])
			zdepths =tf.constant([item.zdepth for item in data])


			###############
			## Setup tf.dataset (Data Feeder)
			###############

			dataset = tf.data.Dataset.from_tensor_slices((imgpaths, zdepths))
			dataset = dataset.map(self.parse_function)
			dataset = dataset.shuffle(self.cfg.buffersize)
			dataset = dataset.batch(self.cfg.batchsize)
			dataset = dataset.repeat()
			return dataset

	def _extend_cfg(self):
		self.cfg.num_batches_per_epoch = math.ceil(self.num_images/self.cfg.batchsize)
		self.cfg.num_steps_per_epoch = self.cfg.num_batches_per_epoch #Because one step is one batch processed
		self.cfg.decay_steps = int(self.cfg.num_epochs_before_decay * self.cfg.num_steps_per_epoch)
		


	def parse_function(self,filename,label):
		image_string = tf.read_file(filename)
		image_decoded = tf.image.decode_jpeg(image_string)
		image_resized = tf.image.resize_images(image_decoded, [224, 224])
		img_float = tf.cast(image_resized,tf.float32)
		avg=tf.constant([[[123.68, 116.779, 103.939]]])
		img=img_float-avg

		return img, tf.expand_dims(tf.cast(label,tf.int32),axis=-1)


