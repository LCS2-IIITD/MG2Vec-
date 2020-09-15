from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy
from models import GAT
# from loss import loss_function

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
adj,adj_L, features, idx_train = load_data()

# Model and optimizer
model = GAT(nfeat=features.shape[1], 
            nhid=args.hidden, 
            dropout=args.dropout, 
            nheads=args.nb_heads, 
            alpha=args.alpha)
optimizer = optim.Adam(model.parameters(), 
                   lr=args.lr, 
                   weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    adj_L = adj_L.cuda()
    idx_train = idx_train.cuda()

features, adj,adj_L = Variable(features), Variable(adj),Variable(adj_L)

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
#     output = model(features, adj)
    loss_train = model(features,adj,adj_L)
    # acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_train.data.item()


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data[0]),
          "accuracy= {:.4f}".format(acc_test.data[0]))

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    
    if epoch%50 == 0:
      loss_values.append(train(epoch))
      torch.save(model.state_dict(), '{}.pkl'.format(epoch))

#       if loss_values[-1] < best:
#           best = loss_values[-1]
#           best_epoch = epoch
#           bad_counter = 0
#       else:
#           bad_counter += 1

#       if bad_counter == args.patience:
#           break

#     files = glob.glob('*.pkl')
#     for file in files:
#         epoch_nb = int(file.split('.')[0])
#         if epoch_nb < best_epoch:
#             os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_ = file.split('.')[0]
    model = GAT(nfeat=features.shape[1], 
            nhid=args.hidden, 
            dropout=args.dropout, 
            nheads=args.nb_heads, 
            alpha=args.alpha)
    model.load_state_dict(torch.load(file))
    model.eval()
    model = model.cuda()
    embeddings = model.get_embedding(features,adj)
    embeddings = embeddings.cpu().detach().numpy()
    print(embeddings.shape,"Numpy array shape")
    np.save(epoch_,embeddings)
    os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
# print('Loading {}th epoch'.format(best_epoch))
# model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))