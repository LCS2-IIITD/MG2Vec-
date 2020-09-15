import json
import pickle
import simplejson
import copy
import time
from scipy.spatial.distance import cosine

def initialise_graph(nnodes,num_layers):

	graph = {}
	for l in range(num_layers):
		graph[l] = {}
		for l1 in range(nnodes):
			graph[l][l1] = []

	return graph


def load_graph(file_name,nnodes,num_features,num_layers):

	graph = initialise_graph(nnodes,num_layers)
	edges = []
	edge_features = []
	f = open(file_name,"r")
	for l in f:
		l = l.strip().split(";")
		l[0],l[1],l[2] = int(l[0]),int(l[1]),int(l[2])
		if l[3] == "edge":
			graph[l[2]][l[0]].append(l[1])
			graph[l[2]][l[1]].append(l[0])

	return graph

def load_node_features(file_name,num_nodes):
	node_features = [[] for i in range(num_nodes)]

	f = open(file_name,"r")
	for l in f:
		temp = l.strip().split(";")
		l = list(map(float,temp[1:]))
		node_features[int(temp[0])] = l

	return node_features


def get_degree(graph):
	degrees = {}
	num_layers = len(graph)
	num_nodes = len(graph[0])
	for l in range(num_layers):
		degrees[l] = []
		for l1 in range(num_nodes):
			degrees[l].append(len(graph[l][l1]))
	return degrees

def initialise_distance(num_nodes):

	return np.zeros(num_nodes)

def get_D_L_set_helper(graph,start_node,k,k1):

	stack = [start_node]
	distance = initialise_distance(len(graph))
	# print(len(distance))
	visited = set()
	temp_l = set()
	temp_d = set()
	counter = 0
	# print(counter)
	while len(stack) != 0:
		node = stack.pop(0)
		visited.add(node)
		# print(len(graph[node]))
		for l in graph[node]:
			if l not in visited:
				distance[l] = distance[node] + 1
				stack.append(l)
				visited.add(l)
		counter += 1
	# print(counter)
	# print(distance)
	count = 0
	for l in range(len(distance)):
		if distance[l] <= k and distance[l] != 0:
			count += 1
			temp_d.add(l)
	# print(count,k,"number in dset")
	#print(len(temp_d),len(temp_l))
	return temp_d

def get_similar_nodes(node_features,threshold,node):
	cur_vector = node_features[node]
	temp_D_set = []
	
	for l in range(len(node_features)):
		if l!=node:
			val = 1 - cosine(cur_vector,node_features[l])
			if val > threshold:
				temp_D_set.append(l)

	return temp_D_set



def get_D_L_set(graph,elig_source,num_nodes,threshold,node_features):
	Dset = []
	source_nodes = []
	counter = 0
	for i in range(len(elig_source)):
		for j in elig_source[i]:
			Dset[i].append(get_similar_nodes(node_features,threshold,j))

	return Dset

def normalize_features(edge_features):

	# print(edge_features)
	edge_features = np.array(edge_features,dtype = float)
	mean = np.mean(edge_features,axis = 0)
	std = np.std(edge_features,axis = 0)
	for l in range(len(edge_features[0])):
		if mean[l] != 0:
			edge_features[:,l] = np.true_divide(edge_features[:,l] - mean[l],std[l])

	return edge_features

if __name__ == '__main__':
	
	print ("Reading data...")
	file_name = 'training_LSE_layer0_EU.txt'
	file_name_NF = 'node_attributes_EU.txt'

	nnodes = 1230
	num_features = 12
	num_layers = 2
	nelig_source = 100
	D_set_thresh = 0.2
	L_set_thresh = 0.2
	total_nodes = nnodes*num_layers
	sim_threshold = 0.9

	graph = load_graph(file_name,nnodes,num_features,num_layers)
	node_features = load_node_features(file_name_NF,nnodes)
	node_features = normalize_features(node_features)
	print("graph loaded")
	degrees = get_degree(graph)
	print("degree done")
	elig_source = [[j for j in range(num_nodes)] for i in range(num_layers)]
	Dset = get_D_L_set(graph,elig_source,nnodes,sim_threshold,node_features)
	f = open("Node_sim_edges.txt","w+")
	for node,l in enumerate(Dset):
		for i in l:
			f.write(str(node)+" "+str(i)+"\n")
	f.close()
	