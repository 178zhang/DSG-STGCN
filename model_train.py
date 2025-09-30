import torch
import torch.nn as nn
from torch.optim import Nadam
from torch.utils.data import DataLoader
from models.dsg_stgcn import DSG_STGCN
from utils.data_loader import get_data_loader
from utils.metrics import compute_metrics
import numpy as np
import matplotlib.pyplot as plt

def combined_loss(y_true, y_pred, alpha=0.1, beta=0.1):
    huber = nn.HuberLoss()(y_true, y_pred)
    cosine_loss = 1 + nn.CosineSimilarity()(y_true, y_pred).mean()
    sparsity_loss = torch.mean(y_pred ** 2) / torch.max(y_pred ** 2 + 1e-8)
    return 1000 * huber + alpha * cosine_loss + beta * sparsity_loss

def train_model(eeg_data, source_data, lead_field, num_epochs=100, batch_size=32, alpha=0.1, learning_rate=0.001, patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DSG_STGCN().to(device)
    optimizer = Nadam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    train_loader = get_data_loader(eeg_data, source_data, lead_field, batch_size=batch_size, shuffle=True)
    
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    early_stop_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for eeg, source, adj, lf in train_loader:
            eeg, source, adj, lf = eeg.to(device), source.to(device), adj.to(device), lf.to(device)
            
            optimizer.zero_grad()
            s_low, s_recon, a_aug = model(eeg, lf)
            
            loss = combined_loss(source, s_recon)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation (using a subset or full test for simplicity; in practice, use separate val data)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for eeg, source, adj, lf in train_loader:  # Replace with val_loader
                eeg, source, adj, lf = eeg.to(device), source.to(device), adj.to(device), lf.to(device)
                s_low, s_recon, a_aug = model(eeg, lf)
                val_loss += combined_loss(source, s_recon).item()
        avg_val_loss = val_loss / len(train_loader)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss) 
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered")
                break
    
    # Plot losses
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()
    
    return model

if __name__ == "__main__":
    # Placeholder data loading (replace with actual data paths)
    eeg_data = np.random.rand(1000, 128, 100)
    source_data = np.random.rand(1000, 2052, 100)
    lead_field = np.random.rand(128, 2052)
    model = train_model(eeg_data, source_data, lead_field)