from easydict import EasyDict as edict
import tensorflow as tf

# cfg for Zdepth classification training

cfg=edict()


cfg.modelfolder='..\\resnet_2000_cont\\'
cfg.imgfolder='..\\imgclass_trainingset\\'
cfg.preprocessed=True
#epsilon = makes labels true for z+-epsilon/2
cfg.epsilon=15

cfg.batchsize=100
cfg.buffersize=5000
cfg.num_classes=2000
cfg.matfilepath = '..\\img_class_png.mat'
cfg.log_dir='..\\logs'
cfg.num_epochs=3000
cfg.initial_learning_rate = 0.05
cfg.learning_rate_decay_factor = 0.97
cfg.num_epochs_before_decay = 5
cfg.global_step = tf.train.get_or_create_global_step()
cfg.locref=False
cfg.locref_loss_weight=0.05
cfg.locrefstdev=7.2801


# add cfg required for getting the output from DLC

cfg_dlc=edict()
