import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
import math
import time
from tensorflow.python.platform import tf_logging as logging
from createdataset import DataHandler
import os

class TFLearn_Resnet:
    def __init__(self, dataset):
        self.dataset = dataset
        self.cfg=self.dataset.cfg
        self._create_batch_iterator()
        self._import_resnet()
        self._setup_tb_summary()

    def _create_batch_iterator(self):
        iterator=self.dataset.data.make_one_shot_iterator()
        if self.cfg.locref:
            self.inputs,self.targets,self.epstarg,self.locref_targets=iterator.get_next()
        else:
            self.inputs,self.targets,self.epstarg=iterator.get_next()

    def _import_resnet(self):
        #Create the model inference
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            self.net, self.end_points = resnet_v1.resnet_v1_50(self.inputs, global_pool=False,
                                                               output_stride=4, is_training=False)

        self.variables_to_restore = slim.get_variables_to_restore(include=['resnet_v1'])
        self.restorer = tf.train.Saver(self.variables_to_restore)

        self.out = self.prediction_layers(self.net, self.end_points)

        #Performs the equivalent to tf.nn.sparse_softmax_cross_entropy_with_logits but enhanced with checks
        self.loss = tf.losses.sigmoid_cross_entropy(self.epstarg,self.out['part_pred'])
        self.total_loss = self.loss

        if self.cfg.locref:
            self.locrefloss=self.cfg.locref_loss_weight*tf.losses.huber_loss(self.locref_targets,
                                                                             self.out['locref'], self.epstarg)
            self.total_loss=self.total_loss+self.locrefloss


        #Define your exponentially decaying learning rate
        self.lr = tf.train.exponential_decay(
            learning_rate = self.cfg.initial_learning_rate,
            global_step = self.cfg.global_step,
            decay_steps = self.cfg.decay_steps,
            decay_rate = self.cfg.learning_rate_decay_factor,
            staircase = True)


        #Now we can define the optimizer that takes on the learning rate
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)

        #Create the train_op.
        self.train_op = slim.learning.create_train_op(self.total_loss, self.optimizer)

    def prediction_layers(self, features, end_points, reuse=None):
        cfg = self.cfg

        out = {}
        with tf.variable_scope('pose', reuse=reuse):
            out['part_pred'] = self.prediction_layer(cfg, features, 'part_pred',
                                                cfg.num_classes)
            if cfg.location_refinement:
                out['locref'] = self.prediction_layer(cfg, features, 'locref_pred',
                                                 cfg.num_classes*2)


        return out

    def prediction_layer(self, cfg, input, name, num_outputs):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], padding='SAME',
                            activation_fn=None, normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(cfg.weight_decay)):
            with tf.variable_scope(name):
                pred = slim.conv2d_transpose(input, num_outputs,
                                             kernel_size=[3, 3], stride=2,
                                             scope='block4')
                return pred


    def _setup_tb_summary(self):
        #Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('losses/Total_Loss', self.loss)
        tf.summary.scalar('learning_rate', self.lr)
        self.my_summary_op = tf.summary.merge_all()

    #Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
    def train_step(self,sess, train_op, global_step):
        '''
        Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
        '''
        #Check the time for each sess run
        start_time = time.time()
        loss, global_step_count = sess.run([train_op, global_step])
        time_elapsed = time.time() - start_time

        #Run the logging to print some results
        logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, loss, time_elapsed)

        return loss, global_step_count

    def restore_fn(self,sess):
        checkpoint_file = self.cfg.checkpoint
        return self.restorer.restore(sess, checkpoint_file)

    def train(self):
        #Now we create a saver function that actually restores the variables from a checkpoint file in a sess
        saver = tf.train.Saver(max_to_keep=5)
        
        #Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir = self.cfg.log_dir, summary_op = None, init_fn = self.restore_fn)

        #Run the managed session
        with sv.managed_session() as sess:
            for step in range(self.cfg.num_steps_per_epoch * self.cfg.num_epochs):
                #At the start of every epoch, show the vital information:
                
                if step % self.cfg.num_batches_per_epoch == 0:
                    logging.info('Epoch %s/%s', step/self.cfg.num_batches_per_epoch + 1, self.cfg.num_epochs)
                    learning_rate_value = sess.run([self.lr])
                    logging.info('Current Learning Rate: %s', learning_rate_value)
                    logging.info('Current Streaming Accuracy: %s')
                    # optionally, print your logits and predictions for a sanity check that things are going fine.
                    print('Epoch: ', step/self.cfg.num_batches_per_epoch)
                    print('Current Learning Rate: \n', learning_rate_value)
                    print('Current Streaming Accuracy: \n')

                loss, _ = self.train_step(sess, self.train_op, sv.global_step)

                #Log the summaries every 10 step.
                if step % 10 == 0:
                    summaries = sess.run(self.my_summary_op)
                    sv.summary_computed(sess, summaries)
                    
                if step % 100==0:
                    print('Step: ', step, 'Epoch: ', math.ceil(step/self.cfg.num_batches_per_epoch))
                    learning_rate_value = sess.run([self.lr])
                    print('Current Loss: \n', loss)
                    print('Current Learning Rate: \n', learning_rate_value)
                    print('Current Streaming Accuracy: \n')

                if step % 50000 == 0:
                    model_name = ".\\img2000cont"
                    saver.save(sess, model_name, global_step=step)

            #We log the final training loss and accuracy
            logging.info('Final Loss: %s', loss)

            #Once all the training has been done, save the log files and checkpoint model
            logging.info('Finished training! Saving model to disk now.')
            saver.save(sess, model_name, global_step=step)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    training=True
    preprocessed = True 
    dataset = DataHandler(training, preprocessed)
    transferlearn=TFLearn_Resnet(dataset)
    transferlearn.train()