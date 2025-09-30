import torch
import torch.nn as nn

class GAE(nn.Module):
    def __init__(self, num_nodes, hidden_dims):
        super(GAE, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dims = hidden_dims
        
        self.gcn1 = nn.Linear(num_nodes, hidden_dims)
        self.gcn2 = nn.Linear(hidden_dims, hidden_dims)
        
    def encode(self, x, adj):
        h = torch.relu(self.gcn1(adj @ x))
        return self.gcn2(adj @ h)
    
    def decode(self, z):
        return torch.sigmoid(torch.matmul(z, z.t()))
    
    def forward(self, x, adj):
        z = self.encode(x, adj) 
        return self.decode(z)