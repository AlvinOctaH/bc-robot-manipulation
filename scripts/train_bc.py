import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

class DemoDataset(Dataset):
    def __init__(self, hdf5_path):
        with h5py.File(hdf5_path, "r") as f:
            obs     = np.array(f["obs"])
            self.actions = torch.FloatTensor(np.array(f["actions"]))

        # Normalisasi: simpan mean dan std untuk dipakai saat evaluate
        self.obs_mean = obs.mean(axis=0)
        self.obs_std  = obs.std(axis=0) + 1e-8  # hindari division by zero
        obs_normalized = (obs - self.obs_mean) / self.obs_std
        self.obs = torch.FloatTensor(obs_normalized)

        print(f"Dataset: {len(self.obs)} timesteps | obs_dim={self.obs.shape[1]} | act_dim={self.actions.shape[1]}")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]


class BCPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh()
        )

    def forward(self, obs):
        return self.net(obs)


def train(data_path="data/demos.hdf5", save_path="results/bc_policy.pth", epochs=200, batch_size=256, lr=1e-3):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = DemoDataset(data_path)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    obs_dim    = dataset.obs.shape[1]
    action_dim = dataset.actions.shape[1]

    policy    = BCPolicy(obs_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"\nTraining BC policy | obs_dim={obs_dim} | action_dim={action_dim}")
    print(f"Epochs={epochs} | Batch={batch_size} | LR={lr}\n")

    best_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0
        for obs_batch, act_batch in loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            pred     = policy(obs_batch)
            loss     = criterion(pred, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch":      epoch,
                "obs_dim":    obs_dim,
                "action_dim": action_dim,
                "obs_mean":   dataset.obs_mean,
                "obs_std":    dataset.obs_std,
                "model_state": policy.state_dict(),
                "loss":       best_loss,
            }, save_path)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] | Loss: {avg_loss:.6f} | Best: {best_loss:.6f}")

    print(f"\nTraining done. Best loss: {best_loss:.6f}")
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train()