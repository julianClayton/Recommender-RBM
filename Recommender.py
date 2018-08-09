from RBM import RBM
import data_reader
import numpy as np
np.set_printoptions(threshold=np.nan)

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

		user = self.dataset.training_X[:,0]

		rbm = RBM(hidden_layer_n=HIDDEN_LAYER_N,iterations=ITERATIONS,dataset=user)

		full_weights = rbm.run()

		print("full_weights")
		print(full_weights.size)




if __name__ == "__main__":
	redcommender = Recommender()