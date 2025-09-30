import torch
import numpy as np

def amplitude_bias(pred, true):
    return torch.mean(torch.abs(pred - true)) / torch.mean(torch.abs(true))

def position_bias(pred_pos, true_pos):
    return torch.mean(torch.norm(pred_pos - true_pos, dim=-1))

def mse_loss(pred, true):
    return torch.mean((pred - true) ** 2)

def compute_metrics(pred_sources, true_sources, pred_positions, true_positions):
    amp_bias = amplitude_bias(pred_sources, true_sources)
    pos_bias = position_bias(pred_positions, true_positions)
    mse = mse_loss(pred_sources, true_sources)
    return {'amplitude_bias': amp_bias.item(), 'position_bias_mm': pos_bias.item(), 'mse': mse.item()} 