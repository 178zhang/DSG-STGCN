import torch
from scipy.signal import hilbert
import numpy as np

def compute_plv(signal):
    # signal: [batch, num_electrodes, time_steps]
    batch_size, num_electrodes, time_steps = signal.shape
    plv = torch.zeros(num_electrodes, num_electrodes).to(signal.device)
    
    for i in range(num_electrodes):
        for j in range(num_electrodes):
            if i != j:
                hilbert_signal = torch.tensor(hilbert(signal[:, i].cpu().numpy(), axis=-1), dtype=torch.complex64).to(signal.device)
                phases = torch.angle(hilbert_signal)
                phase_diff = phases[:, i] - phases[:, j]
                plv[i, j] = torch.abs(torch.mean(torch.exp(1j * phase_diff)))
    
    return plv 