# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 12:58:06 2019

@author: vinayak
"""

#Input edge list, colon seperated
# Layer specific : Degree, clustering coefficient, Betweeness centrality, Closeness centrality
# Also gives graph's : avg clustering coeff, avg path length, edge density/sparsity - used for Dset proximity based selection
import networkx as nx
import csv 
import numpy as np

#Creates the graph. Takes tab separated edge list as input. Undirected unweighted graph.
def createGraph(path,numNodes): 
    with open(path) as file:
        data=csv.reader(file,delimiter=' ')
        F_node=[]
        S_node=[]
        for eachRow in data:
            #print(eachRow[0])
            F_node.append(int(eachRow[0]))
            S_node.append(int(eachRow[1]))
    edgeList=[]        
    for i,j in zip(F_node,S_node):  
        edgeList.append([i,j])
    
    G=nx.Graph()    
    G.add_nodes_from(range(numNodes))
    G.add_edges_from(edgeList)
    #print(G.nodes)
    #print(G.edges)

    return G #An object of networkx class graph

def degree(G):
    deg=np.zeros(len(G.nodes))
    for i in range(len(deg)):
        deg[i]=G.degree[i]
    return deg

def clus_coeff(G):
    clus=np.zeros(len(G.nodes))
    for i in range(len(clus)):
        clus[i]=nx.clustering(G,i)
    return clus

def bw_centrality(G):
    bw_cent=np.zeros(len(G.nodes))
    dum=nx.betweenness_centrality(G)
    for i in range(len(dum)):
        bw_cent[i]=dum[i]    
    return bw_cent

def clos_centrality(G):
    clos_cent=np.zeros(len(G.nodes))
    dum=nx.closeness_centrality(G)
    for i in range(len(dum)):    
        clos_cent[i]=dum[i]    
    return clos_cent

def avg_clus(G):
    return nx.average_clustering(G)

def avg_pathlength(G):
    return nx.average_shortest_path_length(G)

if __name__=='__main__':
    path = "Train_EU_layer0.cites"
    numNodes = 1319
    G = createGraph(path,numNodes)
    print("graph created")
    degrees = degree(G)
    #degrees.dumps()
    clusteringCoefficients = clus_coeff(G)
    #clusteringCoefficients.dumps()
    betweenessCentrality = bw_centrality(G)
    closenessCentrality = clos_centrality(G)
    #closenessCentrality.dumps()
    print("centrality done")
    averageClus = avg_clus(G)
    #averageClus.dumps()
    #averagePathlength = avg_pathlength(G)
        