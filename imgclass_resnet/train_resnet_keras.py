import tensorflow as tf
from tensorflow import keras
from imgreader import DataHandler
import math

## Todo:
# 1. Implement tf.data feeder with batching and epoch (done)
# 2. implement saver (done)
# 3. implement validation code (done)
# 4. implement dynamic learning rate (done)
# 5. make into a class

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
batchsize=100
buffersize=10000

matfilepath = '..\\img_class_data_jpg.mat'
data_handler=DataHandler(matfilepath,batchsize)

# A vector of filenames.
print(len([item.im_path for item in data_handler.data]))
steps_per_epoch = math.ceil(len([item.im_path for item in data_handler.data])/batchsize)
imgpaths=tf.constant([item.im_path for item in data_handler.data])
zdepths =tf.constant([item.zdepth for item in data_handler.data])

# `labels[i]` is the label for the image in `filenames[i].

dataset = tf.data.Dataset.from_tensor_slices((imgpaths, zdepths))

dataset = dataset.map(data_handler.parse_function)
dataset = dataset.repeat()
dataset = dataset.shuffle(buffersize)
dataset = dataset.batch(batchsize)
print(dataset.output_types)
print(dataset.output_shapes)
iterator = dataset.make_one_shot_iterator()

inputs,targets = iterator.get_next()

print(inputs)
print(targets)

def scalarmul(outlayer):
	zdepth=tf.scalar_mul(2000.0,outlayer)
	return zdepth

def huber_loss(y_true, y_pred):
	return tf.losses.huber_loss(y_true,y_pred, weights=1.0, delta=5.0)

def lr_change(epoch):
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

def resnet_inout(inputdata):
	resnet = keras.applications.ResNet50(
	    include_top=False,
	    weights='imagenet',
	    input_tensor=inputdata,
	    input_shape=(256,256,3),
	    pooling=None,
	    classes=1
	)
	resnet.summary()
	out = resnet.get_layer('avg_pool').output
	out = keras.layers.Flatten()(out)
	out = keras.layers.Dense(1, activation="sigmoid")(out)
	prediction = keras.layers.Lambda(scalarmul)(out)
	return prediction

model_input = keras.layers.Input(tensor=inputs)
model_output = resnet_inout(model_input)

resnet_final=keras.Model(inputs=model_input, outputs=model_output)
adam = tf.train.AdamOptimizer(learning_rate=0.0)

resnet_final.compile(optimizer=adam,
              loss=huber_loss,
              metrics=['mae','accuracy'], target_tensors=[targets])

print('compile done!')

resnet_final.summary()


callbacks = [
  # Write TensorBoard logs to `./logs` directory
  keras.callbacks.TensorBoard(log_dir='.\logs_resnet'),
  #dynamic learning rate
  keras.callbacks.LearningRateScheduler(lr_change),
  # save model on check points
  keras.callbacks.ModelCheckpoint('.\\', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=10)
]

resnet_final.fit(epochs=1000,steps_per_epoch=steps_per_epoch,callbacks=callbacks,validation_split=0.1)

