from RBM import RBM
import data_reader
import numpy as np
import os
from pathlib import Path
import tensorflow as tf

np.set_printoptions(threshold=np.nan)

"""
The RBM will create a new RBM for each new user
1. Construct V matrix which has visible softmax units for the user ratings 
2. Pass in weights (and biases??)from the last session
3. Only include non-zero entries 
4. 
"""

HIDDEN_LAYER_N = 30
ITERATIONS 	   = 5
RATING_MAX 	   = 5

WEIGHTS_FILE   		= "data/saved_weights"
HIDDEN_BIAS_FILE 	= "data/saved_hidden_biases"
VISIBLE_BIAS_FILE 	= "data/saved_visible_biases"

class Recommender:

	def __init__(self):

		self.dataset = data_reader.movie_lens()

		self.all_weights = []
		self.all_bh		 = []
		self.all_bv		 = []
		self.w_locations = []

	def train(self):
		#self.dataset.training_X.shape[1]
		self.wb = {}
		for i in range(300):
			user = self.dataset.training_X[:,i]
			rbm = RBM(hidden_layer_n=HIDDEN_LAYER_N,iterations=ITERATIONS,dataset=user)
			rbm.run()

			self.all_weights.append(rbm.full_weights)
			self.all_bv.append(rbm.full_bv)
			self.all_bh.append(rbm.full_bh)

			print("RBM number: " + str(i))

		self.wb[WEIGHTS_FILE] = self.average_matrices(self.all_weights)
		self.wb[VISIBLE_BIAS_FILE] = self.average_matrices(self.all_bv)
		self.wb[HIDDEN_BIAS_FILE] = self.average_matrices(self.all_bh)

		#self.save_matrix(WEIGHTS_FILE)
		#self.save_matrix(VISIBLE_BIAS_FILE)
		#self.save_matrix(HIDDEN_BIAS_FILE)

	def average_matrices(self, M):
		accum_mat = np.zeros(M[0].shape)
		for i in range(len(M)):
			accum_mat += M[i]
		return accum_mat / len(M)


	def save_matrix(self, dataset):
		os.remove(dataset)
		f = open(dataset, "wb")
		np.save(f, self.wb[dataset])

	def load_matrix(self,dataset):
		f = open(dataset, "rb")
		return np.load(f)

	def _gamma(self, x, biases):
		return np.exp(np.dot(x, biases))

	def ratings_to_softmax_units(self, user):
		sm_units = []
		delete_indices = []
		new_weights = self.average_weights
		new_bv  = self.average_bv
		for i in range(len(user)):
			if (user[i] != 0):
				feature = np.zeros(5)
				feature[int(user[i] - 1)] = 1
				for j in range(feature.size):
					sm_units.append(feature[j])
			elif (user[i] == 0):
				[delete_indices.append((5*i) + j) for j in range(5)]
		new_weights = np.delete(new_weights,delete_indices,0)
		features = np.asarray(sm_units)
		return features.flatten().reshape((1, features.size)), new_weights, new_bv

	def recommend(self, u, q):
		self.average_weights = self.load_matrix(WEIGHTS_FILE)
		self.average_bv 	 = self.load_matrix(VISIBLE_BIAS_FILE)
		self.average_bh 	 = self.load_matrix(HIDDEN_BIAS_FILE)
		print(self.dataset.training_X[:,u])
		data, w, bv = self.ratings_to_softmax_units(self.dataset.training_X[:,u])
		print(data.shape)
		print(w.shape)
		print(bv.shape)
		print(data)
		probs = [] 

		#for i in range(RATING_MAX):
		#	for j in range(HIDDEN_LAYER_N):
		#		1 + np.exp(np.dot(user, average_weights) + np.dot(user, average_weights[]))




if __name__ == "__main__":
	recommender = Recommender()
	#recommender.train()
	recommender.recommend(3, 4)



