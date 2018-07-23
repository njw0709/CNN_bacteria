import tensorflow as tf
from imgreader import DataHandler
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
import math
import time
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging

batchsize=100
buffersize=10000
num_classes=1936
matfilepath = '..\\img_class_data_jpg_d.mat'

data_handler=DataHandler(matfilepath,batchsize)
log_dir='..\\logs_resnet'
num_epochs=1000
initial_learning_rate = 0.02
learning_rate_decay_factor = 0.7
num_epochs_before_decay = 2
global_step = tf.train.get_or_create_global_step()
num_batches_per_epoch = math.ceil(len([item.im_path for item in data_handler.data])/batchsize)
num_steps_per_epoch = num_batches_per_epoch #Because one step is one batch processed
decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)
checkpoint_file = '.\\resnet_v1_50.ckpt'



# A vector of filenames.

steps_per_epoch = math.ceil(len([item.im_path for item in data_handler.data])/batchsize)
imgpaths=tf.constant([item.im_path for item in data_handler.data])
zdepths =tf.constant([item.zdepth for item in data_handler.data])

# `labels[i]` is the label for the image in `filenames[i].

dataset = tf.data.Dataset.from_tensor_slices((imgpaths, zdepths))
dataset = dataset.map(data_handler.parse_function)
dataset = dataset.shuffle(buffersize)
dataset = dataset.batch(batchsize)
dataset = dataset.repeat()

iterator = dataset.make_one_shot_iterator()

inputs,targets = iterator.get_next()
print(inputs)
tf.summary.tensor_summary('input',inputs)

#Create the model inference
with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    net, end_points = resnet_v1.resnet_v1_50(inputs, num_classes, is_training=True)

exclude = ['resnet_v1_50/logits','resnet_v1_50/predictions']
variables_to_restore = slim.get_variables_to_restore(exclude = exclude)
 #Perform one-hot-encoding of the labels (Try one-hot-encoding within the load_batch function!)
one_hot_labels = slim.one_hot_encoding(targets, num_classes)
one_hot_labels=tf.expand_dims(one_hot_labels,axis=1)

#Performs the equivalent to tf.nn.sparse_softmax_cross_entropy_with_logits but enhanced with checks
loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits = net)
total_loss = tf.losses.get_total_loss()    #obtain the regularization losses as well




#Define your exponentially decaying learning rate
lr = tf.train.exponential_decay(
    learning_rate = initial_learning_rate,
    global_step = global_step,
    decay_steps = decay_steps,
    decay_rate = learning_rate_decay_factor,
    staircase = True)


#Now we can define the optimizer that takes on the learning rate
optimizer = tf.train.AdamOptimizer(learning_rate = lr)

#Create the train_op.
train_op = slim.learning.create_train_op(total_loss, optimizer)

#State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
predictions = tf.argmax(tf.squeeze(end_points['predictions']), 1)
probabilities = end_points['predictions']
accuracy, accuracy_update = tf.metrics.accuracy(predictions, targets)
metrics_op = tf.group(accuracy_update, probabilities)

#Now finally create all the summaries you need to monitor and group them into one summary op.
tf.summary.scalar('losses/Total_Loss', total_loss)
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('learning_rate', lr)
my_summary_op = tf.summary.merge_all()

#Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
def train_step(sess, train_op, global_step):
    '''
    Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
    '''
    #Check the time for each sess run
    start_time = time.time()
    total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
    time_elapsed = time.time() - start_time

    #Run the logging to print some results
    logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)

    return total_loss, global_step_count

#Now we create a saver function that actually restores the variables from a checkpoint file in a sess
restorer = tf.train.Saver(variables_to_restore)
saver = tf.train.Saver(max_to_keep=5)
def restore_fn(sess):
    return saver.restore(sess, checkpoint_file)

#Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
sv = tf.train.Supervisor(logdir = log_dir, summary_op = None, init_fn = restore_fn)

#Run the managed session
with sv.managed_session() as sess:
    for step in range(num_steps_per_epoch * num_epochs):
        #At the start of every epoch, show the vital information:
        print('Step:', step)
        if step % num_batches_per_epoch == 0:
            logging.info('Epoch %s/%s', step/num_batches_per_epoch + 1, num_epochs)
            learning_rate_value, accuracy_value = sess.run([lr, accuracy])
            logging.info('Current Learning Rate: %s', learning_rate_value)
            logging.info('Current Streaming Accuracy: %s', accuracy_value)
            # optionally, print your logits and predictions for a sanity check that things are going fine.
            logits_value, probabilities_value, predictions_value, labels_value = sess.run([net, probabilities, predictions, targets])
            # print('logits: \n', logits_value)
            # print('Probabilities: \n', probabilities_value)
            # print('predictions: \n', predictions_value)
            # print('Labels:\n:', labels_value)
        if step % num_batches_per_epoch*10 == 0:
            model_name = ".\\imgclass_checkpt"
            saver.save(sess, model_name, global_step=step)

        #Log the summaries every 10 step.
        if step % 10 == 0:
            loss, _ = train_step(sess, train_op, sv.global_step)
            summaries = sess.run(my_summary_op)
            sv.summary_computed(sess, summaries)
            
        #If not, simply run the training step
        else:
            loss, _ = train_step(sess, train_op, sv.global_step)

    #We log the final training loss and accuracy
    logging.info('Final Loss: %s', loss)
    logging.info('Final Accuracy: %s', sess.run(accuracy))

    #Once all the training has been done, save the log files and checkpoint model
    logging.info('Finished training! Saving model to disk now.')
    saver.save(sess, model_name, global_step=step)


