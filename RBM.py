import tensorflow as tf
import numpy as np
import data_reader 
import matplotlib.pyplot as plt
from random import randint
import time


RATING_MAX 	   = 5
class RBM:

	def __init__(
		self,
		hidden_layer_n,
		iterations,
		dataset,
		batch_size=50,
		alpha=0.01,
		W=None,
		training=True):


		self.W = W
		self.batch_size 	= batch_size
		self.iterations 	= iterations
		self.hidden_layer_n = hidden_layer_n

		#We will only use non-zero values when training
		self.full_user		= dataset
		self._full_weights 	= np.zeros((self.full_user.size * RATING_MAX, hidden_layer_n))
		self._full_bh 		= np.zeros((self.hidden_layer_n))
		self._full_bv 		= np.zeros((self.full_user.size * RATING_MAX))

		self.w_locations 	= []
		self.bh_locations	= []
		self.bv_locations	= []	


		#When training the RBM we have to format the input data. When just recommending we use already
		#formatted data. 
		if training:
			self.user 			= self.ratings_to_softmax_units(dataset)
		else:
			self.user= dataset

		num_ratings = 5

		self.input_layer_n 		= self.user.size

		print("full size: " + str(self.full_user.size) + " " + str(self.input_layer_n))

		self.x  = tf.placeholder(tf.float32, [None, self.input_layer_n], name="x") 

		self.W = tf.Variable(tf.random_normal([self.input_layer_n, hidden_layer_n], 0.01), name="W") 


		self.bh = tf.Variable(tf.random_normal([hidden_layer_n], 0.01),  tf.float32, name="bh")

		#This will be a matrix 
		self.bv 	= tf.Variable(tf.random_normal([self.input_layer_n ], 0.01),  tf.float32, name="bv")

		#first pass from input to hidden

		hidden_0 = self.forward_prop(self.x)

		#first pass back to visible
		visible_1  		   = self.get_activations(self.backward_prop(hidden_0))

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
		self.err = tf.reduce_mean(tf.subtract(self.x, self.v_sample)**2)

		self.update_all = [self.W.assign_add(update_weights), self.bv.assign_add(update_bv), self.bh.assign_add(update_bh)]
		self.init  = tf.global_variables_initializer()

	#Staightforward RBM forward propagation 
	def forward_prop(self, visible_samples):
		hidden_activation = tf.nn.sigmoid(tf.matmul(self.x, self.W) + self.bh)
		return self.get_activations(hidden_activation)

	"""
	Probablilties that each individual softmax unit is activated. Since 
	"""
	def backward_prop(self, hidden_samples):
		#had to use some tensorflow wankery to make up for using a 1D vector for the input instead of 2d
		num = tf.exp(tf.matmul(hidden_samples, tf.transpose(self.W))+ self.bv)
		num_reshape = tf.reshape(num, [-1,5])
		dems = tf.reduce_sum(num_reshape, 1)
		probs = tf.transpose(tf.divide(tf.transpose(num_reshape),dems))
		return tf.reshape(probs, [-1, self.user.size])

	def get_activations(self, probs):
		return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

	"""
	While training we only use the movies that the user has rated. After training we need to re-populate the 
	entire weights matrix and the entire visible biases matrix. The locations of the weight and hidden biases are 
	store in the self.w_locations list object. 
	"""
	def _set_full_weights(self, weights):
		for location in self.w_locations:
			for i in range(self.input_layer_n):
				for j in range(self.hidden_layer_n):
					self._full_weights[location][j] = weights[i][j]

	def _set_full_bv(self, biases):
		for location in self.w_locations:
			self._full_bv[location] = location



	def softmax(self, x):
		return np.exp(x) / np.sum(np.exp(x), axis=0)

	"""
	There is some data tomfoolery that needs to be done. For each rating (1-5) we convert it to a one hot
	vector then normalize the vector using softmax. Every un-rated item is removed from the training set but we 
	store their locations in the larger complete weight and bias matrices. To simplify the multiplication we flatten
	the input vector. 
	"""
	def ratings_to_softmax_units(self, user):
		sm_units = []
		for i in range(len(user)):
			if (user[i] != 0):
				feature = np.zeros(RATING_MAX)
				feature[int(user[i] - 1)] = 1
				feature = self.softmax(feature.T)
				for j in range(feature.size):
					sm_units.append(feature[j])
					self.w_locations.append(RATING_MAX*i + j)
		features = np.asarray(sm_units)
		return features.flatten().reshape((1, features.size))

	
	"""
	Using the complete weights from training we sample a given input to make a prediction 
	"""
	def run_prediction(self, data, w, bh, bv):
		self.W 	= tf.convert_to_tensor(w)
		self.bh = tf.convert_to_tensor(bh)
		self.bv = tf.convert_to_tensor(bv)

		init  = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init)
			res = sess.run(self.v_sample, feed_dict={self.x: data})
		return res


	"""
	Training over the dataset. Can take a long time. After training, the averaged weights are used to perform recommendations
	"""
	def run(self):
		with tf.Session() as sess:
			sess.run(self.init)
			iteration = 0
			while iteration < self.iterations:
				sess.run(self.update_all, feed_dict={self.x: self.user})
				new_cost = sess.run(self.err, feed_dict={self.x: self.user})
				sess.run(self.update_all, feed_dict={self.x: self.user})
				res = sess.run([self.x, self.v_sample], feed_dict={self.x: self.user})
				iteration+=1
			self._set_full_weights(self.W.eval())
			self._set_full_bv(self.bv.eval())
			self._full_bh = self.bh.eval()
			print("Done training")

	#Debugging function that displays the input beside the output  
	def display_in_out(self, x, y):
		for i in range(0,self.input_layer_n, RATING_MAX):
			print("=====================")
			for j in range(RATING_MAX):
				print(str(x[0][i+j]) + " " + str(y[0][i+j]))



	@property
	def full_weights(self):
		return self._full_weights

	@property
	def full_bv(self):
		return self._full_bv

	@property
	def full_bh(self):
		return self._full_bh
	
	
	
