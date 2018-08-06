from easydict import EasyDict as edict
import tensorflow as tf

# cfg for Zdepth classification training

cfg=edict()


cfg.modelfolder='..\\imgclass_resnet_huberloss\\'
cfg.imgfolder='..\\imgclass_trainingset\\'
cfg.preprocessed=True

cfg.batchsize=100
cfg.buffersize=10000
cfg.num_classes=1
cfg.matfilepath = '..\\img_class_data_jpg_d.mat'
cfg.log_dir='..\\logs_resnet'
cfg.num_epochs=3000
cfg.initial_learning_rate = 0.05
cfg.learning_rate_decay_factor = 0.96
cfg.num_epochs_before_decay = 5
cfg.global_step = tf.train.get_or_create_global_step()

# add cfg required for getting the output from DLC

cfg_dlc=edict()
