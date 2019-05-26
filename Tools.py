from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import adam, SGD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random


def get_data():
	tmp = np.arange(100)
	x = np.power(tmp, 2)
	y = np.array(tmp) * 100
	r = tmp
	random.shuffle(tmp)
	x = x[r]
	y = y[r]
	x = np.divide(x, 10000)
	y = np.divide(y, 10000)
	print(x)
	print(y)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
	return x, y


def get_weights(model):
	weights = list()
	shapes = list()
	for layer in model.layers:
		for weight in layer.get_weights():
			shapes.append(weight.shape)
			tmp = weight.reshape(-1)
			for i in tmp:
				weights.append(i)
	weights = np.array(weights)
	shapes = np.array(shapes)
	return weights, shapes


def set_weights(model, weights, shapes):
	w_index = 0
	weights_new = list()
	for index, layer in enumerate(shapes):
		s = np.prod(shapes[index])
		tmp_weights = weights[w_index: s]
		weight = tmp_weights.reshape(shapes[index])
		weights_new.append(weight)
	model.set_weights(weights_new)
	return model


def build_model(X, y):
	model = Sequential()
	model.add(Dense(8, input_shape=(1,)))
	model.add(Activation('sigmoid'))
	model.add(Dense(1))
	model.add(Activation('linear'))
	model.summary()
	# weights, shapes = get_weights(model)
	#model = set_weights(model, weights, shapes)
	# model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
	# history = model.fit(X, y, epochs=1000, verbose=1)
	# y_pred = model.predict(X)
	return model


def evaluate_model(model, X, y):
	y_pred = model.predict(X)
	print(y_pred)
	# print(y)
	score = mean_absolute_percentage_error(y, y_pred)
	return score


def mean_absolute_percentage_error(y_true, y_pred):
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
