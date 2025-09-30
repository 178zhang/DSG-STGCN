import torch
import torch.nn as nn

class EAGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(EAGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.W_a = nn.Linear(input_size + hidden_size, input_size)
        self.W_r = nn.Linear(input_size + hidden_size, input_size)
        self.W_u = nn.Linear(input_size + hidden_size, input_size)
        self.W_h = nn.Linear(input_size + hidden_size, input_size)
        
        self.b_a = nn.Parameter(torch.zeros(input_size))
        self.b_r = nn.Parameter(torch.zeros(input_size))
        self.b_u = nn.Parameter(torch.zeros(input_size))
        self.b_h = nn.Parameter(torch.zeros(input_size)) 
        
    def forward(self, x):
        # x: [seq_len * batch, input_size]
        batch_size = x.size(0) // self.hidden_size
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        
        for t in range(x.size(0) // batch_size):
            x_t = x[t * batch_size:(t + 1) * batch_size]
            combined = torch.cat((x_t, h_t), dim=-1)
            
            a_t = torch.sigmoid(self.W_a(combined) + self.b_a)
            r_t = torch.sigmoid(self.W_r(combined) + self.b_r)
            u_t = torch.sigmoid(self.W_u(combined) + self.b_u)
            
            combined_h = torch.cat((x_t, r_t * h_t), dim=-1)
            h_tilde = torch.tanh(self.W_h(combined_h) + self.b_h)
            
            h_t = (1 - u_t) * h_t + u_t * h_tilde
            
        return h_t, h_t