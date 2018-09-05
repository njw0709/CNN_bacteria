import math
import numpy as np
import scipy.io as sio
import tensorflow as tf
import config


class DataItem:
	pass

class DataHandler:
	def __init__(self,istraining,preprocessed):
		self.cfg=config.cfg
		self.istraining=istraining
		if preprocessed:
			self.data = self._load_dataset()
			self._extend_cfg()
		
		

	def _load_dataset(self):
			# Load Matlab file dataset annotation
			mlab = sio.loadmat(self.cfg.matfilepath)
			mlab=mlab['data_compiler']
			self.num_images = mlab.shape[1]
			self.data = []

			for i in range(self.num_images):
				sample = mlab[0, i]

				item = DataItem()
				item.image_id = i
				item.im_path = sample[0][0]
				
				item.positions = sample[1]
				item.scmap = self.compute_scmap(item.positions,[256,256])
				self.data.append(item)

			imgpaths=tf.constant([item.im_path for item in data])
			positions = tf.constant([item.positions for item in data])
			scmaps =tf.constant([item.scmap for item in data])

			###############
			## Setup tf.dataset (Data Feeder)
			###############

			dataset = tf.data.Dataset.from_tensor_slices((imgpaths,positions,scmaps))
			if self.istraining:
				dataset = dataset.map(map_func=self.parse_fn_preprocessed, num_parallel_calls=8)
				dataset = dataset.batch(self.cfg.batchsize)
				dataset = dataset.prefetch(buffer_size=self.cfg.buffersize)
				dataset = dataset.shuffle(self.cfg.buffersize)
				dataset = dataset.repeat()
			else:
				dataset = dataset.map(map_func=self.parse_fn_preprocessed, num_parallel_calls=8)
				dataset = dataset.batch(self.cfg.batchsize)
			return dataset


	def _extend_cfg(self):
		self.cfg.num_batches_per_epoch = math.ceil(self.num_images/self.cfg.batchsize)
		self.cfg.num_steps_per_epoch = self.cfg.num_batches_per_epoch #Because one step is one batch processed
		self.cfg.decay_steps = int(self.cfg.num_epochs_before_decay * self.cfg.num_steps_per_epoch)
		

	def makedataset(self,img,coords):
		dataset = tf.data.Dataset.from_tensor_slices({'img':img, "coords":coords})
		#returns cutout images
		dataset = dataset.map(self.parse_fn_notprocessed)
		dataset = dataset.batch(self.cfg.batchsize)
		return dataset


	def parse_fn_preprocessed(self, imgpath, positions, scmap):
		# Process image
		filename=imgpath
		image_string = tf.read_file(filename)
		image_decoded = tf.image.decode_png(image_string)
		img_float = tf.cast(image_decoded,tf.float32)
		avg=tf.constant([[[123.68, 116.779, 103.939]]])
		img=img_float-avg
		return img, positions, scmap

	def compute_scmap(self,label,img_shape):
		depth = 2000
		scmap=np.zeros([img_shape[0],img_shape[1],depth])
		for bug in label:
			x,y,z=np.meshgrid(np.linspace(-(bug[0]-1),img_shape[0]-(bug[0]-1),img_shape[0]),
							  np.linspace(-(bug[1]-1),img_shape[1]-(bug[1]-1),img_shape[1]),
							  np.linspace(-(bug[2]-1),depth-(bug[0]-1),depth))
			rad_sq=x**2+y**2+z**2
			dist_thresh_sq=self.cfg.pos_thres**2
			bugscmap=(1.0/dist_thresh_sq)*(dist_thresh_sq-rad_sq)
			radmask=rad_sq>dist_thresh_sq
			scmapmask=scmap>0.1
			bugscmap[radmask]=0
			bugscmap[scmapmask]=0
			scmap+=bugscmap
		return scmap




