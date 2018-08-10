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

WEIGHTS_FILE   = "data/saved_weights"

class Recommender:

	def __init__(self):

		self.dataset = data_reader.movie_lens()
		self.all_weights = []

	def train(self):
		for i in range(self.dataset.training_X.shape[1]):
				user = self.dataset.training_X[:,i]
				rbm = RBM(hidden_layer_n=HIDDEN_LAYER_N,iterations=ITERATIONS,dataset=user)
				rbm.run()
				full_weights = rbm.full_weights
				self.all_weights.append(full_weights)

		self.average_weights =  self.average_weights()
		print("total weights: " + str(len(self.all_weights)))
		self.save_weights()

	def average_weights(self):
		accum_mat = np.zeros(self.all_weights[0].shape)
		for i in range(len(self.all_weights)):
			accum_mat += self.all_weights[i]
		return accum_mat / len(self.all_weights)

	def save_weights(self):
		os.remove(WEIGHTS_FILE)
		f = open(WEIGHTS_FILE, "wb")
		np.save(f, self.average_weights)

	def load_weights(self):
		f = open(WEIGHTS_FILE, "rb")
		return np.load(f)


if __name__ == "__main__":
	redcommender = Recommender()
	redcommender.train()