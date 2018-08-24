from RBM import RBM
import data_reader
import numpy as np
import os
from pathlib import Path
import tensorflow as tf
import math 

np.set_printoptions(threshold=np.nan)

"""
The RBM will create a new RBM for each new user
1. Construct V matrix which has visible softmax units for the user ratings 
2. Pass in weights (and biases??)from the last session
3. Only include non-zero entries 
4. 
"""

HIDDEN_LAYER_N = 50
ITERATIONS 	   = 100
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

		#wb is a dictionary that stores the average Weight matrices and Bias matrices. The keys are where the 
		#files are stored 
		self.wb = {}



		"""
		For each user an RBM will be created. 
		"""
		for i in range(self.dataset.training_X.shape[1]):
			user = self.dataset.training_X[:,i]
			rbm = RBM(hidden_layer_n=HIDDEN_LAYER_N,iterations=ITERATIONS,dataset=user)
			rbm.run()

			#After the an RBM is run the weights and biases are re-added to complete  set. 
			self.all_weights.append(rbm.full_weights)
			self.all_bv.append(rbm.full_bv)
			self.all_bh.append(rbm.full_bh)

			print("RBM number: " + str(i))


		#Average all the weights and all the biases from all the RBM's (With each RBM corresponding to a user)
		self.wb[WEIGHTS_FILE] = self.average_matrices(self.all_weights)
		self.wb[VISIBLE_BIAS_FILE] = self.average_matrices(self.all_bv)
		self.wb[HIDDEN_BIAS_FILE] = self.average_matrices(self.all_bh)


		#Training can take a long time so we can save the weights and biases 
		self.save_matrix(WEIGHTS_FILE)
		self.save_matrix(VISIBLE_BIAS_FILE)
		self.save_matrix(HIDDEN_BIAS_FILE)

	#Average all the matrices in a list 
	def average_matrices(self, M):
		accum_mat = np.zeros(M[0].shape)
		for i in range(len(M)):
			accum_mat += M[i]
		return accum_mat / len(M)


	def save_matrix(self, dataset):
		os.remove(dataset)
		f = open(dataset, "wb")
		np.save(f, self.wb[dataset])

	def _load_matrix(self,dataset):
		f = open(dataset, "rb")
		return np.load(f)


	def softmax(self, x):
		return np.exp(x) / np.sum(np.exp(x), axis=0)


	"""
	Convert all the one-hot rating to softmax units. 
	There is a similar method to this in the RBM class. However this method 
	"""
	def _ratings_to_softmax_units(self, user, q):
		sm_units = []
		delete_indices = []
		new_weights = self.average_weights
		new_bv  = self.average_bv
		for i in range(len(user)):
			if (user[i] != 0):
				feature = np.zeros(5)
				feature[int(user[i] - 1)] = 1
				feature = self.softmax(feature.T)
				[sm_units.append(feature[j]) for j in range(RATING_MAX)]
			elif (i == q):
				#This is a placeholder for the predicted movie so it does not get removed 
				[sm_units.append(-1) for j in range(RATING_MAX)]
			elif (user[i] == 0):
				[delete_indices.append((RATING_MAX*i) + j) for j in range(RATING_MAX)]
		new_weights = np.delete(new_weights,delete_indices,0)
		new_bv		= np.delete(new_bv, delete_indices, 0)
		features = np.asarray(sm_units)
		return features.flatten().reshape((1, features.size)), new_weights, new_bv

	#When we re-size the weight and biases the query item will have a new index 
	def _new_index(self, data):
		for i in range(data.size):
			if data[0][i] == -1:
				for j in range(RATING_MAX):
					data[0][i + j] = 0
				return data, i 


	#Set the rating in a matrix (one-hot). Evey 5 units corresponds to a single rating 
	def _set_rating(self, data, index, rating):
		for i in range(data.size):
			if i == index:
				for j in range(RATING_MAX):
					data[0][i + j] = 0
				data[0][i + (rating - 1)] = 1
		return data


	#Gets the rating at query q 
	def _get_rating(self,data, q):
		rating = []
		for j in range(RATING_MAX):
			rating.append(data[0][q + j])
		return np.asarray(rating)



	#retrieves each individual probabilty for each rating
	def _get_probabilities(self, results, q):
		probs = []
		for i in range(len(results)):
			rating = self._get_rating(results[i], q)
			probs.append(rating[i])
		return probs

	#Calculates the expeted value from a list of probabilities 
	def _expected_value(self, p):
		accum = 0
		for i in range(len(p)):
			accum += (i+1)*p[i]
		return accum

	"""
	This method gets a little complicated and I should probably come back and clean it up when I get time
	"""
	def recommend(self, u, q):
		self.average_weights = self._load_matrix(WEIGHTS_FILE)
		self.average_bv 	 = self._load_matrix(VISIBLE_BIAS_FILE)
		self.average_bh		 = self._load_matrix(HIDDEN_BIAS_FILE)

		#x is the input (the user we are predicting for), w is the weights of just the items rated by user x and bv is 
		# just the visible biases of the corresponding to the items rated by the user
		x, w, bv 		 = self._ratings_to_softmax_units(self.dataset.training_X[:,u], q)
		#bh is the same size for every user because they correspond to the hidden units
		bh = self.average_bh

		#Returns the new index for the query q. The index changes because we reomved all unrated items.
		#The units for the query are changed to 5 -1s as a placeholder 
		x, new_q  = self._new_index(x)
		results = []

		#For each rating 1-5 we sample the RBM with the corresponding input.
		for r in range(RATING_MAX):
			x = self._set_rating(x, new_q, (r+1))
			rbm = RBM(hidden_layer_n=HIDDEN_LAYER_N,iterations=ITERATIONS,dataset=x, training=False)
			sample = rbm.run_prediction(x, w, bh, bv)
			results.append(sample)

		#Get the expected output from each of the probablitlies of each input 
		probs = self._get_probabilities(results, new_q)
		prediction = self._expected_value(self.softmax(probs))

		print("Prediction: user " + str(u) + " will give movie: " + str(q) + " rating: " + str(prediction))

if __name__ == "__main__":
	recommender = Recommender()
	recommender.recommend(40, 346)


