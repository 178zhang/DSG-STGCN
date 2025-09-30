import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_source_reconstruction(pred_sources, true_sources, title="Source Reconstruction"):
    plt.figure(figsize=(10, 5))
    plt.plot(true_sources.cpu().numpy().T, label='True')
    plt.plot(pred_sources.cpu().numpy().T, label='Predicted', alpha=0.7)
    plt.title(title)
    plt.legend()
    plt.show()

def plot_biases(metrics):
    plt.figure(figsize=(10, 5))
    plt.bar(['Amplitude Bias', 'Position Bias (mm)'], [metrics['amplitude_bias'], metrics['position_bias_mm']])
    plt.title('Localization Biases')
    plt.ylabel('Value')
    plt.show()

def plot_brain_map(sources, title="Brain Activity Map"):
    plt.figure(figsize=(10, 5))
    plt.imshow(sources.cpu().numpy(), aspect='auto')
    plt.title(title)
    plt.colorbar(label='Amplitude')
    plt.show() 