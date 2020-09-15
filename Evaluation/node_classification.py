
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

if __name__ == '__main__':

	X = np.load("450.npy")
	node_mapping = {}
	labels = []
	X_val = []
	f = open("cora.content","r")
	for l in f:
		l = l.strip().split("\t")
		labels.append(l[-1])
	labels_dic = {}
	counter = 1
	for l in labels:
		if l not in labels_dic:
			labels_dic[l] = counter
			counter += 1
	labels = [labels_dic[l] for l in labels]

	X_train, X_test, y_train, y_test = train_test_split(X, labels,stratify=labels,test_size=0.1)
	clf = LogisticRegression()
	model = clf.fit(X_train,y_train)
	print(accuracy_score(model.predict(X_test),y_test),"accuracy_score")