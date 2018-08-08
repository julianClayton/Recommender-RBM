from RBM import RBM
import data_reader
import numpy as np
"""
The RBM will create a new RBM for each new user
1. Construct V matrix which has visible softmax units for the user ratings 
2. Pass in weights (and biases??)from the last session
3. Only include non-zero entries 
4. 
"""

HIDDEN_LAYER_N = 30
ITERATIONS 	   = 100
RATING_MAX 	   = 5

class Recommender:

	def __init__(self):

		self.dataset = data_reader.movie_lens()

		user = self.ratings_to_softmax_units(self.dataset.training_X[:,0])

		rbm = RBM(hidden_layer_n=HIDDEN_LAYER_N,iterations=ITERATIONS,user=user.flatten().reshape((1, user.size)))

		rbm.run()


	def softmax(self, x):
		return np.exp(x) / np.sum(np.exp(x), axis=0)

	def ratings_to_softmax_units(self, user):
		sm_units = []
		for rating in user:
			if (rating != 0):
				feature = np.zeros(5)
				feature[int(rating - 1)] = 1
				sm_units.append(self.softmax(feature.T))
		return np.asarray(sm_units)




if __name__ == "__main__":
	redcommender = Recommender()