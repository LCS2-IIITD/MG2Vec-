import numpy as np

class generateLset(object):
  def __init__(self,graph):
    self.graph = graph
    self.degrees = np.sum(self.graph,axis=2)
    print(self.degrees)

  def getLsetnode(self,node,graph,layer):
    a = np.array([i for i in range(self.graph.shape[1])])
    a = [i for i in a if graph[node][i]!=1]
    temp_degree = np.array([self.degrees[layer][i] for i in a])
    temp_degree = temp_degree/np.sum(temp_degree)
    n_sample = int(self.degrees[layer][node])
    try:
      if n_sample !=0:
        temp_Lset = np.random.choice(a,p=temp_degree,size=n_sample)
        return temp_Lset
      else:
        print(1)
        return []
    except:
      print(2)
      return a
  
  def getLsetlayer(self,layer):
    Lset = []
    graph = self.graph[layer]
    for index in range(graph.shape[0]):
      Lset.append(self.getLsetnode(index,graph,layer))
    return Lset
  
  def getLset(self):
    Lset = []
    for layer in range(len(self.graph)):
      Lset.append(self.getLsetlayer(layer))
    return Lset

def read_graph(file_name,layers,num_nodes):
  f = open(file_name,"r")
  graph = [np.zeros((num_nodes,num_nodes)) for i in range(layers)]
  for l in f:
    l = list(map(int,l.strip().split(" ")))
    graph[l[2]][l[0]][l[1]] = 1
    graph[l[2]][l[1]][l[0]] = 1
  return np.array(graph)

def main():
  num_layer = 2
  num_nodes = 1320
  file_name = "EU_train_layer_0.txt"
  num_sample = 20
  graph = read_graph(file_name,num_layer,num_nodes)
  Lsetobj = generateLset(graph)
  Lset = Lsetobj.getLset()
  print(Lset[0][0])
  for layer,lset in enumerate(Lset):
    f = open("EU_train_layer_0_Lset_"+str(layer)+".txt","w+")
    for node,no_edges in enumerate(lset):
      for no_edge in no_edges:
        f.write(str(node)+" "+str(no_edge)+"\n")
    f.close()
  graph = graph.tolist()
  for layer,lset in enumerate(graph):
    f = open("EU_train_layer_0_"+str(layer)+".txt","w+")
    for node,no_edges in enumerate(lset):
      for index,no_edge in enumerate(no_edges):
        if no_edge == 1:
          f.write(str(node)+" "+str(index)+"\n")
    f.close()  

  



main()