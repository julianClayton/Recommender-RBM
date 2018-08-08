import pandas as pd
from sklearn import cross_validation as cv
import numpy as np


def read_data():
	header = ['user_id', 'item_id', 'rating', 'timestamp']
	df = pd.read_csv('data/ml-100k/u.data', sep='\t', names=header)

	n_users = df.user_id.unique().shape[0]
	n_items = df.item_id.unique().shape[0]
	print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))


	train_data, test_data = cv.train_test_split(df, test_size=0.25)

	train_data_matrix = np.zeros((n_users, n_items))
	for line in train_data.itertuples():
	    train_data_matrix[line[1]-1, line[2]-1] = line[3]

	test_data_matrix = np.zeros((n_users, n_items))
	for line in test_data.itertuples():
	    test_data_matrix[line[1]-1, line[2]-1] = line[3]
	print(train_data_matrix.T.shape)
	return train_data_matrix.T, test_data_matrix.T


	