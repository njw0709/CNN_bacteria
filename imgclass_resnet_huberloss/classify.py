"""
Written by: Jong Woo Nam for 3d bacteria tracking purpose

jwnam0709@gmail.com

To use: 
Instantiate the Classify class (input = true / false)
- for validation of the network, pass in True. 
- for prediction on the data, pass in False.

Then, do Classify.initialize()
Finally, iterate over the dataset by making a for loop and computing predictions or accuracies.


"""

####################################################
# Dependencies
####################################################

import sys
import os
from createdataset import DataHandler
from config import cfg

class Classify:
    def __init__(self,cfg,testing):
        self.cfg=cfg
        self.dh=DataHandler(False,self.cfg.preprocessed)
        self._create_batch_iterator(testing)
        self._load_pretrainednet(testing)


    def _create_batch_iterator(self,testing):
        iterator=self.dh.data.make_one_shot_iterator()
        if testing:
            self.inputs, self.answers =iterator.get_next()
        else:
            self.inputs, _ =iterator.get_next()


    def _load_pretrainednet(self,testing):

        # Check which snap shots are available and get the most recent version

        Snapshots = np.array([
            fn.split('.')[0]
            for fn in os.listdir(modelfolder)
            if "index" in fn
        ])
        increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
        Snapshots = Snapshots[increasing_indices]
        self.cfg['init_weights'] = self.cfg.modelfolder + Snapshots[snapshotindex]

        #setup the network for prediction

        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            self.net, self.end_points = resnet_v1.resnet_v1_50(self.inputs, self.cfg.num_classes, is_training=False)

        zerotoone = tf.sigmoid(self.net)
        self.predictions=tf.squeeze(tf.scalar_mul(2000.0,zerotoone),axis=[1,2])
        if testing:
            self.accuracy = tf.metrics.accuracy(self.predictions, self.answers)
        
    def initialize(self):
        #restore from the disk and initialize a session
        restorer = tf.train.Saver()

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        # Restore variables from disk.
        restorer.restore(sess, self.cfg.init_weights)

    def predictZ(self):
        #get predictions
        Zpositions=self.sess.run(self.predictions)
        return Zpositions

    def getaccuracy(self):
        Accuracy = self.sess.run(self.accuracy)
        return Accuracy



