import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.softmax = torch.nn.Softmax(dim=1)
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
    
    def loss_function(self,output,target,target_L):
        target_D = target
        loss_D = -1 * target_D * torch.log(torch.sigmoid(torch.mm(output,torch.t(output))))
        loss_L = torch.tensor([-1 * target_L * torch.log(1 - torch.sigmoid(torch.mm(output,torch.t(output))))])
        loss_matrix = torch.sum([torch.mean(torch.abs(loss_D*target.shape[0] + loss)) for loss in loss_L])
        loss_regularization = torch.sum(torch.tensor([torch.mean(torch.abs(att.W)) for att in self.attentions])) + torch.sum(torch.tensor([torch.mean(torch.abs(att.a)) for att in self.attentions]))
        return loss_matrix * 10 + loss_regularization
      
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        print(x.size())
        x = x.view(x.size()[0],self.nheads,-1)
        x = torch.mean(x,dim = 1)
        x = self.softmax(x)
        return self.loss_function(x,adj)
    
    def get_embedding(self,x,adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        print(x.size())
        x = x.view(x.size()[0],self.nheads,-1)
        x = torch.mean(x,dim = 1)
        softmax = torch.nn.Softmax(dim=1)
        x = softmax(x)
        return x