import torch
import numpy as np
from models.dsg_stgcn import DSG_STGCN
from utils.data_loader import get_data_loader
from utils.metrics import compute_metrics
from utils.visualize import plot_source_reconstruction, plot_biases

def test_synthetic(eeg_data, source_data, lead_field, snrs=[0, 5, 10, 15, 20], source_configs=["isolated", "extended"]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DSG_STGCN().to(device)
    model.eval()
    
    results = {}
    for snr in snrs:
        for config in source_configs:
            # Simulate SNR and config effect (placeholder)
            noisy_eeg = eeg_data + np.random.normal(0, 10**(-snr/20), eeg_data.shape)
            train_loader = get_data_loader(noisy_eeg, source_data, lead_field, batch_size=32)
            
            total_metrics = {'amplitude_bias': 0, 'position_bias_mm': 0, 'mse': 0}
            num_batches = 0
            
            with torch.no_grad():
                for eeg, source, adj, lf in train_loader:
                    eeg, source, adj, lf = eeg.to(device), source.to(device), adj.to(device), lf.to(device)
                    s_low, s_recon, _ = model(eeg, lf)
                    metrics = compute_metrics(s_recon, source, torch.zeros_like(source), torch.zeros_like(source))
                    for key in total_metrics:
                        total_metrics[key] += metrics[key]
                    num_batches += 1
            
            avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
            results[f"SNR_{snr}_{config}"] = avg_metrics
            print(f"SNR {snr}dB, {config}: {avg_metrics}")
            plot_source_reconstruction(s_recon[0], source[0], f"SNR {snr}dB {config}")
            plot_biases(avg_metrics)
    
    return results

if __name__ == "__main__":
    eeg_data = np.random.rand(1000, 128, 100)
    source_data = np.random.rand(1000, 2052, 100)
    lead_field = np.random.rand(128, 2052)
    results = test_synthetic(eeg_data, source_data, lead_field) 