from easydict import EasyDict as edict
import tensorflow as tf

# cfg for Zdepth classification training

cfg=edict()


cfg.modelfolder='..\\resnet_2000_dense\\'
cfg.imgfolder='..\\imgclass_trainingset\\'
cfg.preprocessed=True
#epsilon = makes labels true for z+-epsilon/2
cfg.epsilon=15
cfg.net_type='resnet_50'
cfg.weight_decay = 0.0001
cfg.batchsize=100
cfg.buffersize=200
cfg.num_classes=2000
cfg.matfilepath = '..\\img_class_data_multilabel.mat'
cfg.log_dir='..\\logs'
cfg.num_epochs=3000
cfg.initial_learning_rate = 0.05
cfg.learning_rate_decay_factor = 0.97
cfg.num_epochs_before_decay = 5
cfg.global_step = tf.train.get_or_create_global_step()
cfg.locref=False
cfg.locref_loss_weight=0.05
cfg.locrefstdev=3.0
cfg.pos_thres=10
cfg.checkpoint='.\img2000cont-500000'

# add cfg required for getting the output from DLC

cfg_dlc=edict()
