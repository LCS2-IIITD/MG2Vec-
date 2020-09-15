import random
layer = 0
f_out = open("Train_EU_layer0.cites","w+")
edges = []
f = open("final_final.txt","r")
for l in f:
	l = l.strip().split(";")
	if int(l[2]) == layer and l[3]=="edge" and l[0]!=l[1]:
		# f_out.write(l[0]+" "+l[1]+"\n")
		edges.append(l[0]+" "+l[1])

random.shuffle(edges)
train_size = int(0.8*len(edges))
train_edges = edges[0:train_size]
test_edges = edges[train_size:]

for i in train_edges:
	f_out.write(i+"\n")
f_out = open("Test_EU_layer0.cites","w+")
for i in test_edges:
	f_out.write(i+"\n")
