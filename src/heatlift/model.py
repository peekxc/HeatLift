import os

import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["TORCH"] = torch.__version__
os.environ["DGLBACKEND"] = "pytorch"

try:
	import dgl

	installed = True
except ImportError:
	installed = False
	print("DGL installed!" if installed else "DGL not found!")


## Graph Convolutional Layer
class GCNLayer(nn.Module):
	def __init__(self, in_size, out_size):
		super(GCNLayer, self).__init__()
		self.W = nn.Linear(in_size, out_size)

	def forward(self, A, X):
		I = dglsp.identity(A.shape)
		A_hat = A + I
		D_hat = dglsp.diag(A_hat.sum(0))
		D_hat_invsqrt = D_hat**-0.5
		return D_hat_invsqrt @ A_hat @ D_hat_invsqrt @ self.W(X)


## Graph Convolutional Network == stacked GCNLayers
class GCN(nn.Module):
	def __init__(self, in_size, out_size, hidden_size):
		super(GCN, self).__init__()
		self.conv1 = GCNLayer(in_size, hidden_size)
		self.conv2 = GCNLayer(hidden_size, out_size)

	def forward(self, A, X):
		X = self.conv1(A, X)
		X = F.relu(X)
		return self.conv2(A, X)


def evaluate_acc(pred: torch.tensor, labels: torch.tensor, masks: dict):
	"""Compute accuracy on validation/test set."""
	val_mask, test_mask = masks["val"], masks["test"]
	val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
	test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
	return val_acc, test_acc


def train_gcn(model, features: torch.tensor, labels: torch.tensor, masks: dict, G: dict = None, **kwargs):
	train_mask = masks["train"]
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
	loss_fun = nn.CrossEntropyLoss()

	## Prep adjacency matrix
	if G is not None:
		N = G["num_nodes"]
		A = dglsp.spmatrix(G["edges"], shape=(N, N))

	for epoch in range(kwargs.get("epochs", 400)):
		model.train()
		logits = model(A, features) if G is not None else model(features)
		loss = loss_fun(logits[train_mask], labels[train_mask])

		## Backward.
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		## Compute prediction
		pred = logits.argmax(dim=1)

		# Evaluate the prediction.
		val_acc, test_acc = evaluate_acc(pred, labels, masks)
		if epoch % 5 == 0:
			print(f"In epoch {epoch}, loss: {loss:.3f}, val acc: {val_acc:.3f}" f", test acc: {test_acc:.3f}")


import time
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F

from dhg import Graph, Hypergraph
from dhg.data import Cora
from dhg.models import HGNN
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
