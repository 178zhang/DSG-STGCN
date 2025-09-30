import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from .plv import compute_plv

class EEGSourceDataset(Dataset):
    def __init__(self, eeg_data, source_data, lead_field):
        self.eeg_data = torch.FloatTensor(eeg_data)
        self.source_data = torch.FloatTensor(source_data)
        self.lead_field = torch.FloatTensor(lead_field)
        
    def __len__(self):
        return self.eeg_data.size(0)
    
    def __getitem__(self, idx):
        eeg = self.eeg_data[idx]
        source = self.source_data[idx]
        adj = compute_plv(eeg.unsqueeze(0))  # Dynamic graph via PLV
        return eeg, source, adj, self.lead_field

def get_data_loader(eeg_data, source_data, lead_field, batch_size=32, shuffle=True):
    dataset = EEGSourceDataset(eeg_data, source_data, lead_field)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) 