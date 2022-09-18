"""GCN using DGL nn package
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
# from dgl.nn import GraphConv
from .graphconv_edge_weight import GraphConvEdgeWeight as GraphConv

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 normalization='none'):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation, norm=normalization))

        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation, norm=normalization))

        self.layers.append(GraphConv(n_hidden, n_classes, norm=normalization))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features, g, edge_weight):
        h = features
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h, edge_weights=edge_weight)

        return h

class MHSA2(nn.Module):
  def __init__(self, num_heads, dim):
    super().__init__()
    self.q = nn.Linear(dim, dim)
    self.k = nn.Linear(dim, dim)
    self.v = nn.Linear(dim, dim)
    self.num_heads = num_heads

  def forward(self, x):
    B, N, C = x.shape
    # 生成转换矩阵并分多头
    q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    k = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    v = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

    attn = q @ k.transpose(2, 3) * (x.shape[-1] ** -0.5)
    attn = attn.softmax(dim=-1)

    v = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
    return v

