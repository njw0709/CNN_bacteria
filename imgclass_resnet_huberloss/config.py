from easydict import EasyDict as edict
import tensorflow as tf

cfg=edict()

cfg.batchsize=100
cfg.buffersize=10000
cfg.num_classes=1
cfg.matfilepath = '..\\img_class_data_jpg_d.mat'
cfg.log_dir='..\\logs_resnet'
cfg.num_epochs=1000
cfg.initial_learning_rate = 0.02
cfg.learning_rate_decay_factor = 0.8
cfg.num_epochs_before_decay = 2
cfg.global_step = tf.train.get_or_create_global_step()
cfg.log_dir= '.\\logs_resnet'