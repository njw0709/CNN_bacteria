import tensorflow as tf

from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops

import tensorflow.contrib.losses as tf_losses
from imgclass_dataset import DataHandler

training_epochs=1000000
batch_size=100
name = "Convnet_imgclass"
datapath='..\\img_class_data.mat'
train_proportion=0.85

## Todo: Setup batch feeding / batch normalization (done)
## divide up training set and test set (done)
## implement shuffle function (done)
## modify the network structure (done)
## make it into a class (done)
## try training / debug (done)

## more todo:
## setup for tensorboard (done)
## save midpoint training results (done)
## train for 1M epochs (done)
## changed learning rates for different epochs (done)

## final todo: 
## implement regularization to prevent overfitting

class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def huber_loss(self, labels, predictions, weight=1.0, k=1.0, scope=None):
        """Define a huber loss  https://en.wikipedia.org/wiki/Huber_loss
          tensor: tensor to regularize.
          k: value of k in the huber loss
          scope: Optional scope for op_scope.

        Huber loss:
        f(x) = if |x| <= k:
                  0.5 * x^2
               else:
                  k * |x| - 0.5 * k^2

        Returns:
          the L1 loss op.

        http://concise-bio.readthedocs.io/en/latest/_modules/concise/tf_helper.html
        """
        with ops.name_scope(scope, "absolute_difference",
                            [predictions, labels]) as scope:
            predictions.get_shape().assert_is_compatible_with(labels.get_shape())
            if weight is None:
                raise ValueError("`weight` cannot be None")
            predictions = math_ops.to_float(predictions)
            labels = math_ops.to_float(labels)
            diff = math_ops.subtract(predictions, labels)
            abs_diff = tf.abs(diff)
            losses = tf.where(abs_diff < k,
                              0.5 * tf.square(diff),
                              k * abs_diff - 0.5 * k ** 2)
            return tf.losses.compute_weighted_loss(losses, weight)



    def _build_net(self):
        with tf.variable_scope(name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.training = tf.placeholder(tf.bool)
            self.keep_prob=tf.placeholder(tf.float32)
            self.learning_rate=tf.placeholder(tf.float32)

            # img 256x256x1 (black/white), Input Layer
            self.X_img = tf.placeholder(tf.float32, [None, 256, 256, 1])
            self.Zdepth = tf.placeholder(tf.float32, [None, 1])

            bn1 = tf.layers.batch_normalization(inputs=self.X_img, training=self.training)


            # Outputs: 
            # Conv1: 256x256x32
            # Pool1: 128x128x32

            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(inputs=bn1, filters=32, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout1 = tf.layers.dropout(inputs=pool1,
                                         rate=0.3, training=self.training)

            bn2 = tf.layers.batch_normalization(inputs=dropout1, training=self.training)


            # Outputs: 
            # Conv2: 128x128x64
            # Pool2: 64x64x64
                    
            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(inputs=bn2, filters=64, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2,
                                         rate=0.3, training=self.training)
            bn3 = tf.layers.batch_normalization(inputs=dropout2, training=self.training)

            # Outputs: 
            # Conv3: 64x64x128
            # Pool3: 32x32x128
            
            # Convolutional Layer #3 and Pooling Layer #3
            conv3 = tf.layers.conv2d(inputs=bn3, filters=128, kernel_size=[3, 3],
                                     padding="same", activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
                                            padding="same", strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3,
                                         rate=0.3, training=self.training)

            bn4 = tf.layers.batch_normalization(inputs=dropout3, training=self.training)

            # Outputs: 
            # Conv4: 32x32x256
            # Pool4: 16x16x256
            
            # Convolutional Layer #4 and Pooling Layer #4
            conv4 = tf.layers.conv2d(inputs=bn4, filters=256, kernel_size=[3, 3],
                                     padding="same", activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
            pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2],
                                            padding="same", strides=2)
            dropout4 = tf.layers.dropout(inputs=pool4,
                                         rate=0.3, training=self.training)

            bn5 = tf.layers.batch_normalization(inputs=dropout4, training=self.training)


            # Outputs: 
            # Conv5: 16x16x512
            # Pool5: 8x8x512
            
            # Convolutional Layer #5 and Pooling Layer #5
            conv5 = tf.layers.conv2d(inputs=bn5, filters=512, kernel_size=[3, 3],
                                     padding="same", activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
            pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2],
                                            padding="same", strides=2)
            dropout5 = tf.layers.dropout(inputs=pool5,
                                         rate=0.3, training=self.training)

            bn6 = tf.layers.batch_normalization(inputs=dropout5, training=self.training)

            # Outputs: 
            # Conv6: 8x8x1024
            # Pool6: 4x4x1024
            
            # Convolutional Layer #6 and Pooling Layer #6
            conv6 = tf.layers.conv2d(inputs=bn6, filters=1024, kernel_size=[3, 3],
                                     padding="same", activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
            pool6 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2],
                                            padding="same", strides=2)
            dropout6 = tf.layers.dropout(inputs=pool6,
                                         rate=0.3, training=self.training)

            bn7 = tf.layers.batch_normalization(inputs=dropout6, training=self.training)


            # Dense Layer with Relu
            flat = tf.reshape(dropout6, [-1, 1024 * 4 * 4])
            dense7 = tf.layers.dense(inputs=flat,
                                     units=625, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            dropout7 = tf.layers.dropout(inputs=dense7,
                                         rate=0.5, training=self.training)

            bn8 = tf.layers.batch_normalization(inputs=dropout7, training=self.training)


            # Sigmoid: L7 Final FC 625 inputs -> 1 output
            dense8 = tf.layers.dense(inputs=bn8, units=1)
            zerotoone = tf.sigmoid(dense8)
            self.predictions = zerotoone*2000.0


        self.cost = self.huber_loss(self.Zdepth, self.predictions,weight=1.0, k=50.0)
        cost_summ = tf.summary.scalar("cost",self.cost)
        #update code for batch normalization during training!
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(tf.cast(self.predictions,tf.uint8)
            ,tf.cast(self.Zdepth,tf.uint8))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.predictions,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def get_lr(self,epoch):
        # [0.005, 10000]
        # [0.02, 430000]
        # [0.002, 730000]
        # [0.001, 1030000]
        if epoch<10000:
            lr=0.005
        elif epoch>=10000 and epoch< 430000:
            lr=0.02
        elif epoch>=430000:
            lr=0.002

        return lr


    def train(self, x_data, y_data, tb_summary,epoch,training=True):
        return self.sess.run([self.cost, self.optimizer, tb_summary], feed_dict={
            self.X_img: x_data, self.Zdepth: y_data, self.learning_rate: self.get_lr(epoch),self.training: training})

if __name__ == '__main__':
    # initialize
    sess=tf.Session()
    m1 = Model(sess, "m1")
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('.\\logs\\2dto3d_logs')
    writer.add_graph(sess.graph)
    datahandler = DataHandler(datapath,batch_size)
    sess.run(tf.global_variables_initializer())

    print('Learning Started!')
    
    datahandler.make_train_test_set(train_proportion)
    saver = tf.train.Saver(max_to_keep=5)

    step=0
    # train my model
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = datahandler.numbatch
        for i in range(total_batch):
            batch_xs, batch_ys = datahandler.get_nextbatch_fortraining()
            c, _, summary = m1.train(batch_xs, batch_ys, merged_summary,epoch)
            writer.add_summary(summary,global_step=step)
            avg_cost += c / total_batch
            step+=1
            print('batchnum:','%04d' %(i+1), 'cost =', '{:.9f}'.format(c))

        print('Epoch:', '%06d' % (epoch + 1), 'avg_cost =', '{:.9f}'.format(avg_cost))

        if epoch % 10 == 0:
            model_name = ".\\imgclass_checkpt"
            saver.save(sess, model_name, global_step=step)

    print('Learning Finished!')

    # Test model and check accuracy
    test_img,test_zdepth=datahandler.retrieve_xydata(datahandler.testset)
    print('Accuracy:', m1.get_accuracy(test_img, test_zdepth))