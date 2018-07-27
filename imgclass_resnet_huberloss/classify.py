"""
Written by: Jong Woo Nam for 3d bacteria tracking purpose

jwnam0709@gmail.com

To use: 
Instantiate the Classify class (input = true / false)
- for validation of the network, pass in True. 
- for prediction on the data, pass in False.

Then, do Classify.initialize()
Finally, iterate over the dataset by making a for loop and computing predictions or accuracies.

Example code (Validation):

imgclass=Classify(True)
imgclass.initialize()
for iteration in range(1,int(img_num_total/batch_size)):
    zpred = imgclass.predictZ()
    acc = imgclass.getaccuracy()

"""

####################################################
# Dependencies
####################################################

import sys
import os
from createdataset import DataHandler
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1


class Classify:
    def __init__(self,cfg,testing):
        self.dh=DataHandler(False,cfg.preprocessed)
        self.cfg = self.dh.cfg
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
            for fn in os.listdir(self.cfg.modelfolder)
            if "index" in fn
        ])
        increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
        Snapshots = Snapshots[increasing_indices]
        self.cfg['init_weights'] = self.cfg.modelfolder + Snapshots[-1]
        print(self.cfg['init_weights'])

        #setup the network for prediction

        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            self.net, self.end_points = resnet_v1.resnet_v1_50(self.inputs, self.cfg.num_classes, is_training=False)

        ## method - Huber loss
        # zerotoone = tf.sigmoid(self.net)
        # self.predictions=tf.squeeze(tf.scalar_mul(2000.0,zerotoone),axis=[1,2])
        # if testing:
        #     self.accuracy = tf.metrics.accuracy(self.predictions, self.answers)
        
        ## method - 2000 classes
        #Perform one-hot-encoding of the labels (Try one-hot-encoding within the load_batch function!)
        one_hot_labels = slim.one_hot_encoding(self.answers, self.cfg.num_classes)
        one_hot_labels=tf.expand_dims(one_hot_labels,axis=1)
        
        self.predictions = tf.argmax(tf.squeeze(self.end_points['predictions']), 1)
        if testing:
            self.accuracy, accuracy_update = tf.metrics.accuracy(self.predictions, self.answers)


    def initialize(self):
        #restore from the disk and initialize a session
        restorer = tf.train.Saver()
        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        # Restore variables from disk.
        restorer.restore(self.sess, self.cfg.init_weights)

    def predictZ(self):
        #get predictions
        Zpositions=self.sess.run(self.predictions)
        return Zpositions

    def getaccuracy(self):
        Pred = self.sess.run(self.predictions)
        Accuracy = self.sess.run(self.accuracy)
        Answers = self.sess.run(self.answers)

        return Accuracy, Pred, Answers


