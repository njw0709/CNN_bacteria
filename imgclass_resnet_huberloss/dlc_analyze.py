import tensorflow as tf
from config import cfg
from classify import Classify
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

testing = True
resnettest = Classify(cfg,testing)
resnettest.initialize()
for idx in range(1,resnettest.cfg.num_batches_per_epoch):
	acc, pred, ans=resnettest.getaccuracy()
	print(acc)
	print(pred)
	print(ans)



