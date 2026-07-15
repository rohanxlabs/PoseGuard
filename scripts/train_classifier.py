"""
Train InjuryMLP on annotated pose feature data.
Expected annotation format: outputs/logs/injury_report.json
"""
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.injury_mlp import InjuryMLP
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


def load_features(json_path: str):
    """
    Load pre-extracted feature vectors saved during pipeline run.
    Each entry: {"features": [...], "label": 0 or 1}
    """
    with open(json_path) as f:
        data = json.load(f)
    X = np.array([d["features"] for d in data], dtype=np.float32)
    y = np.array([d["label"] for d in data], dtype=np.int64)
    return X, y


def train(data_path: str = "data/annotations/features.json",
          save_path: str = "models/injury_mlp.pt",
          epochs: int = 50, lr: float = 1e-3, batch_size: int = 32):

    X, y = load_features(data_path)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pickle.dump(scaler, open("models/scaler.pkl", "wb"))

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=batch_size, shuffle=True
    )

    model = InjuryMLP(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            xv = torch.tensor(X_val)
            preds = model(xv).argmax(dim=1).numpy()
            acc = (preds == y_val).mean()
        print(f"Epoch {epoch+1:3d}/{epochs} | Val Acc: {acc:.3f}")

        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), save_path)

    print(f"\nBest val accuracy: {best_val_acc:.3f} | Saved to {save_path}")


if __name__ == "__main__":
    train()