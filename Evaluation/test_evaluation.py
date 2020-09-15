#Evaluation

import numpy as np
import simplejson
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score,precision_score,recall_score,accuracy_score,f1_score
import csv
import pickle
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

def find_precision(model,X_test,Y_test):
    return precision_score(model.predict(X_test),Y_test)

def find_recall(model,X_test,Y_test):
    return recall_score(model.predict(X_test),Y_test)

def find_auc(model,pro,Y_test):
    return roc_auc_score(Y_test,pro)

def find_accuracy(model,X_test,Y_test):
    return  accuracy_score(model.predict(X_test),Y_test)

def find_f1_score(model,X_test,Y_test):
    return  f1_score(Y_test,model.predict(X_test))


if __name__ == '__main__':
    
#     dic = np.load("embeddings.npy")
#     dic = np.load("embeddings_average.npy")
    dic = np.load("161.npy")
    count_nodes = 1319
    folder = "./"
    X = []
    Y = []
    
    node_mapping = {}
    counter = 0
    f = open("node_features_Hike.txt","r")
    for l in f:
        l = l.strip().split(" ")
        node_mapping[l[0]] = counter
        counter += 1


    print("Number of Embeddings",len(dic),len(dic[0]))
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    
    print(node_mapping)
    edges = list(csv.reader(open(folder+"Hike_layer_0.txt","r"),delimiter=' '))
    for index,edge in enumerate(edges):
        if node_mapping[edge[0]] > node_mapping[edge[1]]:
            edges[index] = [node_mapping[edge[0]],node_mapping[edge[1]]]
        else:
            edges[index]= [node_mapping[edge[1]],node_mapping[edge[0]]]
    
    edges = set([str(edge[0])+" "+str(edge[1]) for edge in edges])
    print(list(edges)[0])
    print("edges formed")
    all_edges = set([str(l)+" "+str(l1) for l in range(count_nodes) for l1 in range(l+1,count_nodes)])
    print(list(all_edges)[0])
    print("all edges formed")
    all_noedge = list(all_edges.difference(edges))
    
    print(len(all_edges),len(edges),len(all_noedge),"Size of edges and no edges")
    
    noedges_train,noedges_test = train_test_split(all_noedge,shuffle=True)
    print(noedges_test[0],noedges_train[0])
    noedges_train = [ [int(l1) for l1 in l.strip().split(" ") ] for l in noedges_train ]
    noedges_test = [ [int(l1) for l1 in l.strip().split(" ") ] for l in noedges_test ]
    
    
    edges_train = list(csv.reader(open(folder+"Hike_train_layer_0_0.txt","r"),delimiter=' '))
    edges_test = list(csv.reader(open(folder+"Hike_test_layer_0.txt","r"),delimiter=' '))
    
    print(len(edges_train),len(noedges_train),len(edges_test),len(noedges_test),"Size of different datasets")
    
    class_train_edge = []
    class_train_noedge = []
    
    for l in edges_train:
        class_train_edge.append((dic[node_mapping[l[0]]]+dic[node_mapping[l[1]]])/2)
    for l in noedges_train:
        class_train_noedge.append((dic[int(l[0])]+dic[int(l[1])])/2)

    X_train.extend(class_train_edge)
    Y_train.extend([1 for i in range(len(class_train_edge))])

    X_train.extend(class_train_noedge)
    Y_train.extend([0 for i in range(len(class_train_noedge))])



    class_test_edge = []
    class_test_noedge = []

    for l in edges_test:
        class_test_edge.append((dic[node_mapping[l[0]]]+dic[node_mapping[l[1]]])/2)
    for l in noedges_test:
        class_test_noedge.append((dic[int(l[0])]+dic[int(l[1])])/2)

    
    X_test.extend(class_test_edge)
    Y_test.extend([1 for i in range(len(class_test_edge))])


    X_test.extend(class_test_noedge)
    Y_test.extend([0 for i in range(len(class_test_noedge))])

    print(len(X_train),len(Y_train),"Number Of Samples")

    print(len(X_test),len(Y_test),"Number Of Samples")

    clf = LogisticRegression()
    model = clf.fit(X_train,Y_train)
        
    print("Train Done")
    print("precision",find_precision(model,X_test,Y_test))
    print("recall",find_recall(model,X_test,Y_test))
    print("f1 score",find_f1_score(model,np.transpose(np.dot(model.coef_,np.transpose(np.array(X_test)))),Y_test))
    print("accuracy",find_accuracy(model,X_test,Y_test))
    print("auc-roc",find_auc(model,X_test,Y_test))
