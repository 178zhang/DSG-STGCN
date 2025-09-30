import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import hilbert
import numpy as np
from .eagru import EAGRU
from .gae import GAE

class DSG_STGCN(nn.Module):
    def __init__(self, num_electrodes=128, num_voxels=2052, time_steps=100, hidden_dims=64, dct_dim=512):
        super(DSG_STGCN, self).__init__()
        self.num_electrodes = num_electrodes
        self.num_voxels = num_voxels
        self.time_steps = time_steps
        self.hidden_dims = hidden_dims
        self.dct_dim = dct_dim 
        
        # Graph construction via PLV
        self.plv_threshold = 0.5  # Placeholder, adjust based on median + std
        
        # Graph Data Augmentation (GAE)
        self.gae = GAE(num_electrodes, hidden_dims)
        
        # Spatiotemporal Feature Extractor (2-layer GCN + EAGRU)
        self.gcn1 = nn.Linear(num_electrodes, hidden_dims)
        self.gcn2 = nn.Linear(hidden_dims, hidden_dims)
        self.eagru = EAGRU(input_size=hidden_dims, hidden_size=hidden_dims, num_layers=1)
        
        # DCT/IDCT for dimension reduction
        self.dct_matrix = self._build_dct_matrix(num_voxels, dct_dim)
        
    def _build_dct_matrix(self, N, K):
        dct_m = np.zeros((K, N))
        for k in range(K):
            for n in range(N):
                if k == 0:
                    dct_m[k, n] = 1.0 / np.sqrt(N)
                else:
                    dct_m[k, n] = np.sqrt(2.0 / N) * np.cos(np.pi * (2 * n + 1) * k / (2 * N))
        return torch.FloatTensor(dct_m)
    
    def _compute_plv(self, z):
        # Hilbert transform for phase
        z_hilbert = torch.tensor(hilbert(z.cpu().numpy(), axis=-1), dtype=torch.complex64).to(z.device)
        phases = torch.angle(z_hilbert)
        plv = torch.zeros(self.num_electrodes, self.num_electrodes).to(z.device)
        for i in range(self.num_electrodes):
            for j in range(self.num_electrodes):
                phase_diff = phases[:, i] - phases[:, j]
                plv[i, j] = torch.abs(torch.mean(torch.exp(1j * phase_diff)))
        return plv
    
    def _graph_augmentation(self, z, a):
        # GAE for connection probability
        p = self.gae(z, a)
        # Interpolation with original adjacency
        rho = 0.5  # Hyperparameter, adjust as needed
        e = rho * p + (1 - rho) * a
        # Relaxed Bernoulli sampling
        tau = 0.1  # Temperature parameter
        gumbel = -torch.log(-torch.log(torch.rand_like(e)))
        a_prime = torch.sigmoid((torch.log(e) - torch.log(1 - e) + gumbel) / tau)
        return a_prime
    
    def forward(self, z, lead_field):
        # z: [batch, num_electrodes, time_steps]
        batch_size = z.size(0)
        
        # Graph construction
        a = self._compute_plv(z)
        a = (a >= self.plv_threshold).float()
        
        # Graph data augmentation
        a_aug = self._graph_augmentation(z, a)
        d = torch.sum(a_aug, dim=-1)
        d_inv_sqrt = torch.pow(d, -0.5).unsqueeze(-1)
        a_norm = d_inv_sqrt * a_aug * d_inv_sqrt.transpose(-2, -1)
        
        # Spatiotemporal feature extraction
        h = F.relu(self.gcn1(a_norm @ z.transpose(1, 2)).transpose(1, 2))
        h = F.relu(self.gcn2(a_norm @ h.transpose(1, 2)).transpose(1, 2))
        h = h.transpose(1, 2).contiguous().view(batch_size * self.time_steps, self.hidden_dims)
        h, _ = self.eagru(h)
        h = h.view(batch_size, self.time_steps, self.hidden_dims).transpose(1, 2)
        
        # DCT for dimension reduction
        s = torch.matmul(lead_field, z.transpose(1, 2))  # [batch, num_voxels, time_steps]
        s_dct = torch.matmul(self.dct_matrix, s)  # [batch, dct_dim, time_steps]
        s_low = s_dct[:, :self.dct_dim, :]  # Low-frequency subspace
        
        # IDCT for reconstruction (optional, for loss)
        s_recon = torch.matmul(self.dct_matrix.transpose(0, 1), s_low)
        
        return s_low, s_recon, a_aug