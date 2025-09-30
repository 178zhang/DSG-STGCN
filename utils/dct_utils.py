import torch
import numpy as np
from scipy.fftpack import dct, idct

def apply_dct(data, dct_dim):
    data_np = data.cpu().numpy()
    dct_out = np.zeros((data_np.shape[0], dct_dim, data_np.shape[2]))
    for i in range(data_np.shape[0]):
        dct_out[i] = dct(data_np[i], axis=0, norm='ortho')[:dct_dim]
    return torch.FloatTensor(dct_out)

def apply_idct(data, full_dim):
    data_np = data.cpu().numpy()
    idct_out = np.zeros((data_np.shape[0], full_dim, data_np.shape[2]))
    for i in range(data_np.shape[0]):
        idct_out[i] = idct(data_np[i], axis=0, norm='ortho', n=full_dim)
    return torch.FloatTensor(idct_out) 