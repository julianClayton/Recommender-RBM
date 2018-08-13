from RBM import RBM
import data_reader
import numpy as np
import os
from pathlib import Path

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


	def train(self):
		#self.dataset.training_X.shape[1]
		self.wb = {}
		for i in range(10):
			user = self.dataset.training_X[:,i]
			rbm = RBM(hidden_layer_n=HIDDEN_LAYER_N,iterations=ITERATIONS,dataset=user)
			rbm.run()

			self.all_weights.append(rbm.full_weights)
			self.all_bv.append(rbm.full_bh)
			self.all_bh.append(rbm.full_bv)

			print("RBM number: " + str(i))

		self.wb[WEIGHTS_FILE] = self.average_matrices(self.all_weights)
		self.wb[VISIBLE_BIAS_FILE] = self.average_matrices(self.all_bv)
		self.wb[HIDDEN_BIAS_FILE] = self.average_matrices(self.all_bh)

		self.save_matrix(WEIGHTS_FILE)
		self.save_matrix(VISIBLE_BIAS_FILE)
		self.save_matrix(HIDDEN_BIAS_FILE)

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

	def _gamma(self):
		print("todo")

	def recommend(self):
		self.average_weights = self.load_weights()


if __name__ == "__main__":
	redcommender = Recommender()
	redcommender.train()