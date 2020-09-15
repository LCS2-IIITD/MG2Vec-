import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_training_curve(X,Y,ylabel,xlabel,legends,title,filename):

	plt.figure()
	colors = ['aqua', 'darkorange', 'cornflowerblue',"blue","green","red","cyan","magenta","yellow","black","brown"]
	n_classes = 2
	
	for i in range(len(Y)):
		plt.plot(X, Y[i], color = colors[i],label = legends[i])

	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.legend(loc="lower right")
	plt.savefig(filename)
	plt.show()

def plot_classification(X,Y):
	plt.figure()
	plt.bar(X,Y)
	plt.xlabel('Methods', fontsize=5)
	plt.ylabel('Accuracy', fontsize=5)
	plt.xticks(X, X, fontsize=5, rotation=30)
	plt.title('Node Classification')
	plt.savefig("Node_Classification.png")
	plt.show()


if __name__ == '__main__':

	# Y =  [[0.76,0.77,0.76,0.8,0.84,0.88,0.813,0.813,0.88,0.87],[0.67,0.76,0.81,0.83,0.85,0.88,0.89,0.89,0.88,0.86],[0.79,0.83,0.84,0.86,0.87,0.88,0.88,0.86,0.86,0.85]]
	# X = [20,40,60,80,100,120,140,160,180,200]
	# xlabel = "Epochs"
	# ylabel = "Accuracy"
	# legends = ["EU","Hike","Lazega"]
	# title = "Accuracy vs Number Of Epochs"
	# plot_training_curve(X,Y,ylabel,xlabel,legends,title,"Epochs.png")
	
	# Y = [[0.67,0.83,0.87,0.9,0.86],[0.63,0.74,0.81,0.89,0.89],[0.78,0.86,0.91,0.88,0.81]]
	# X = [2,4,6,8,10]
	# xlabel = "Number of Attention Heads"
	# ylabel = "Accuracy"
	# legends = ["EU","Hike","Lazega"]
	# title = "Accuracy vs Number Of Attention Heads"
	# plot_training_curve(X,Y,ylabel,xlabel,legends,title,"Attention.png")

	# Y = [[0.82,0.86,0.88,0.9,0.9],[0.78,0.83,0.89,0.89,0.87],[0.88,0.9,0.91,0.91,0.88]]
	# X = [16,32,64,128,256]
	# xlabel = "Embedding Dimension"
	# ylabel = "Accuracy"
	# legends = ["EU","Hike","Lazega"]
	# title = "Accuracy vs Embedding Dimension"
	# plot_training_curve(X,Y,ylabel,xlabel,legends,title,"Dimension.png")

	Y = [0.846,0.795,0.817,0.834,0.795,0.815]
	X = ["Multigraph2vec++","PMNE","Watch Your Step","MNE","Node2vec","ASNE"]
	plot_classification(X,Y)