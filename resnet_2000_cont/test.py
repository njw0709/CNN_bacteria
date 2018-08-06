import tensorflow as tf
depth=20
label=[1,3,4,6,7,8,9]
label_onehot=tf.one_hot(label,depth)
epsilon=10

kernel=tf.ones([1,epsilon],dtype=tf.float32)
kernel=tf.transpose(kernel)
kernel=tf.expand_dims(kernel,axis=-1)
kernel=tf.expand_dims(kernel,axis=-1)

label_onehot1=tf.transpose(label_onehot)
label_onehot1=tf.expand_dims(label_onehot1,axis=0)
label_onehot1=tf.expand_dims(label_onehot1,axis=-1)

print(kernel)
print(label_onehot1)
output = tf.nn.conv2d(label_onehot1,kernel,strides=[1,1,1,1],padding="SAME")
output = tf.squeeze(output,[0,-1])
output = tf.transpose(output)
print(output)
score = tf.constant([[0.5, 0.6, 0.2, 0.01], 
                     [0.8, 0.75, 1.0, 1.0]])

maxind=tf.argmax(score,dimension=1)
maxind=tf.one_hot(maxind,4)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	orig=sess.run(label_onehot)
	print(orig)
	b=sess.run(output)
	print(b)
	c=sess.run(maxind)
	print(c)
	