import tensorflow as tf
import numpy as np
import data_reader 
import matplotlib.pyplot as plt
from random import randint


MNIST = "MNIST"

class RBM:

	def __init__(
		self,
		hidden_layer_n,
		iterations,
		user,
		batch_size=50,
		alpha=0.01,
		W=None):


		self.W = W
		self.batch_size 	= batch_size
		self.display_step 	= 3
		self.iterations 	= iterations

		self.user 			= user
		num_ratings 		= 5

		input_layer_n 		= self.user.size
		print("shape: " + str(self.user.shape))
		print("input_layer_n: " + str(input_layer_n))
		#print(self.user)


		self.x  = tf.placeholder(tf.float32, [None, input_layer_n], name="x") 

		self.W = tf.Variable(tf.random_normal([input_layer_n, hidden_layer_n], 0.01), name="W") 


		self.bh 	= tf.Variable(tf.random_normal([hidden_layer_n], 0.01),  tf.float32, name="bh")

		#This will be a matrix 
		self.bv 	= tf.Variable(tf.random_normal([input_layer_n ], 0.01),  tf.float32, name="bv")


		#first pass from input to hidden

		hidden_0 = self.forward_prop(self.x)


		#first pass back to visible
		visible_1  		   = self.backward_prop(hidden_0)

		#second pass back to hidden 
		hidden_1 = self.forward_prop(visible_1)

		#Gradients for weights
		self.postive_grad 	= tf.matmul(tf.transpose(self.x), hidden_0)
		self.negative_grad 	= tf.matmul(tf.transpose(visible_1), hidden_1)

		update_weights 		= alpha * tf.subtract(self.postive_grad, self.negative_grad)
		update_bv 			= alpha * tf.reduce_mean(tf.subtract(self.x, visible_1), 0)
		update_bh 			= alpha * tf.reduce_mean(tf.subtract(hidden_0 , hidden_1), 0)

		h_sample = self.forward_prop(self.x)

		self.v_sample = self.backward_prop(h_sample)

		#variable v is the probabilities of the visible sample (not the states). This is nicer for visualization
		self.v = tf.nn.sigmoid(tf.matmul(h_sample, tf.transpose(self.W)) + self.bv)
		self.err = tf.reduce_mean((self.x - self.v_sample)**2)

		self.update_all = [self.W.assign_add(update_weights), self.bv.assign_add(update_bv), self.bh.assign_add(update_bh)]


		self.init  = tf.global_variables_initializer()

	def forward_prop(self, visible_samples):
		hidden_activation = tf.nn.sigmoid(tf.matmul(self.x, self.W) + self.bh)
		return self.get_activations(hidden_activation)

	def backward_prop(self, hidden_samples):
		self.num = tf.exp(tf.matmul(hidden_samples, tf.transpose(self.W))+ self.bv)
		self.dem = tf.reduce_sum(self.num)

		self.prob = tf.divide(self.num,self.dem)
		self.new_shape = tf.reshape(self.num, [-1,5])
		self.dems = tf.reduce_sum(self.new_shape, 1)
		probs = tf.divide(tf.transpose(self.new_shape),self.dems)
		return self.get_activations(tf.reshape(probs, [-1, self.user.size]))

	def get_activations(self, probs):
		return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))


	def softmax(self, x):
		return np.exp(x) / np.sum(np.exp(x), axis=0)

	def run(self):
		with tf.Session() as sess:
			sess.run(self.init)
			diff = 100
			cost = 0
			iteration = 0
			while iteration < 2000:
				sess.run(self.update_all, feed_dict={self.x: self.user})
				new_cost = sess.run(self.err, feed_dict={self.x: self.user})
				#print(sess.run(self.v_sample, feed_dict={self.x: self.user}))
				#print(sess.run(self.num, feed_dict={self.x: self.user}))
				#print(sess.run(self.new_shape, feed_dict={self.x: self.user}))				
				iteration+=1
				if (iteration % self.display_step == 0):
					print("iteration: " + str(iteration)+  " /" + str(self.iterations) + " COST: " + str(new_cost))
			print("converged after: " + str(iteration) + " iterations")

if __name__ == "__main__":
	rbm = RBM(500,100, dataset=MOVIE_LENS)
	rbm.run()