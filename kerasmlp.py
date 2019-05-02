import argparse
import csv
import os
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True'


"""
Function to plot the decision regions for each model, one with contours and hard boundaries, another as a heatmap type plot
"""
def plot_decision_regions(model, trainsize, units, include_offset, save=False, resolution=0.01):

    cmap = ListedColormap(('red', 'blue'))

    # plot the decision surface
    xmin, xmax = -0.55, 1.55
    x, y = np.meshgrid(np.arange(xmin, xmax, resolution), np.arange(xmin, xmax, resolution))
    if include_offset:
    	preds = model.predict(np.array([x.ravel(), y.ravel(), np.array([-1] * len(x.ravel()))]).T)
    else:
    	preds = model.predict(np.array([x.ravel(), y.ravel()]).T)
    preds = preds.reshape(x.shape)
    plt.clf()
    plt.contourf(x, y, preds, alpha=0.5, cmap=cmap, levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.scatter(x=[0,0,1,1], y=[0,1,0,1], marker='+', alpha=0.75, c='black')
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    legend_elements = [Patch(facecolor='red', edgecolor='red', label='0'), Patch(facecolor='blue', edgecolor='blue', label='1')]
    plt.legend(handles=legend_elements)
    if save:
        plt.savefig('plots/contour' + str(trainsize) + '+' + str(units))
    else:
        plt.show()
    
    plt.clf()
    plt.imshow(preds, cmap='Spectral', interpolation='nearest', vmin=0, vmax=1, extent=[-0.55, 1.55, 1.55, -0.55])
    plt.gca().invert_yaxis()
    plt.colorbar()
    if save:
        plt.savefig('plots/heat' + str(trainsize) + '+' + str(units))
    else:
        plt.show()


"""
Plot training & validation loss values
"""
def plot_loss(history, trainsize, units, save=False):

	plt.clf()
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	if save:
		plt.savefig('plots/loss' + str(trainsize) + '+' + str(units))
	else:
		plt.show()


"""
Utilises the Gaussian distribution to create an array of inputs and targets for training the models
"""
def create_data(trainsize, stddev, include_offset):

	L = int(trainsize / 4)
	N = 4 * L
	if include_offset:
		X = np.matrix([
			np.array([0, 0, 1, 1] * L) + np.random.normal(0, stddev, N), 
			np.array([0, 1, 0, 1] * L) + np.random.normal(0, stddev, N),
			np.array([-1] * N)]).T
	else:
		X = np.matrix([
			np.array([0, 0, 1, 1] * L) + np.random.normal(0, stddev, N), 
			np.array([0, 1, 0, 1] * L) + np.random.normal(0, stddev, N)]).T
	y = np.array([0, 1, 1, 0] * L)

	return X, y


"""
Defines the architecture of the model, a three-layer perceptron
"""
def create_model(units, learnrate, include_offset):

	model = Sequential()
	if include_offset:
		model.add(Dense(units, input_dim=3, use_bias=False))
	else:
		model.add(Dense(units, input_dim=2))
	model.add(Activation('sigmoid'))
	if include_offset:
		model.add(Dense(1, use_bias=False))
	else:
		model.add(Dense(1))
	model.add(Activation('sigmoid'))
	
	# Could also test different optimizers???
	model.compile(loss='mean_squared_error', optimizer=SGD(lr=learnrate))

	return model


def main():

	parser = argparse.ArgumentParser(description='3-Layer MLP to solve XOR problem')
	parser.add_argument('-o', '--offset', action="store_true", help='Include offset bias vector in model inputs')
	parser.add_argument('-u', '--units', default=4, type=int, help='Number of units in the middle hidden layer of the network')
	parser.add_argument('-n', '--trainsize', default=16, type=int, help='Number of items in the training set, USE A MULTIPLE OF 4')
	parser.add_argument('-t', '--testsize', default=64, type=int, help='Number of items in the testing set, USE A MULTIPLE OF 4')
	parser.add_argument('-e', '--epochs', default=32768, type=int, help='Number of training epochs (number is divided by trainsize)')
	parser.add_argument('-s', '--stddev', default=0.5, type=float, help='Standard deviation of Gaussian noise')
	parser.add_argument('-lr', '--learnrate', default=0.1, type=float, help='SGD learning rate')
	parser.add_argument('-a', '--all', action="store_true", help='Automatically try all combinations required for the assignment')
	args = parser.parse_args()
	print()
	if args.offset:
		print('Including offset vector in model inputs.')
		print()

	if args.all:

		for trainsize, units in [(16, 2), (16, 4), (16, 8), (32, 2), (32, 4), (32, 8), (64, 2), (64, 4), (64, 8)]:
			
			X, y = create_data(trainsize, args.stddev, args.offset)
			valX, valy = create_data(trainsize, args.stddev, args.offset)
			model = create_model(units, args.learnrate, args.offset)

			history = model.fit(X, y, validation_data=(valX, valy), batch_size=1, epochs=int(args.epochs/trainsize), verbose=0)

			plot_decision_regions(model, trainsize, units, args.offset, save=True)
			plot_loss(history, trainsize, units, save=True)

			testX, testy = create_data(args.testsize, args.stddev, args.offset)
			preds = model.predict_proba(testX)
			mse = np.square(testy - preds.ravel()).mean()
			avg_error = abs(testy - preds.ravel()).mean()
			print('trainsize: ' + str(trainsize) + ', units: ' + str(units) + ', MSE: ' + str(mse) + ', avg error: ' + str(avg_error))

			if args.offset:
				with open('avg_errors_offset.csv', 'a') as file:
					writer = csv.writer(file)
					writer.writerow([trainsize, units, mse, avg_error])
			else:
				with open('avg_errors.csv', 'a') as file:
					writer = csv.writer(file)
					writer.writerow([trainsize, units, mse, avg_error])

	else:

		X, y = create_data(args.trainsize, args.stddev, args.offset)
		model = create_model(args.units, args.learnrate, args.offset)

		history = model.fit(X, y, batch_size=1, epochs=int(args.epochs/args.trainsize), verbose=1)
		print(np.column_stack([y, model.predict_proba(X)]))
		print(model.summary())

		plot_decision_regions(model, args.trainsize, args.units, args.offset)
		plot_loss(history, args.trainsize, args.units)

		testX, testy = create_data(args.testsize, args.stddev, args.offset)
		preds = model.predict_proba(testX)
		mse = np.square(testy - preds.ravel()).mean()
		avg_error = abs(testy - preds.ravel()).mean()


if __name__ == "__main__":
	main()