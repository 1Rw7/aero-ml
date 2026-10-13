"""训练脚本：运行一次即可生成 models/aero_net.pth"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from core.data_gen import generate_data, normalize
from core.model import AeroNet

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "aero_net.pth"
)


def train(n_epochs=100, batch_size=256, verbose=True):
    X, y = generate_data(n_samples=20000)
    X_norm, y_norm, stats = normalize(X, y)

    n_train = int(len(X_norm) * 0.8)
    X_train = torch.tensor(X_norm[:n_train])
    y_train = torch.tensor(y_norm[:n_train])
    X_val = torch.tensor(X_norm[n_train:])
    y_val = torch.tensor(y_norm[n_train:])

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AeroNet(hidden=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(X_train)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(device))
            val_loss = loss_fn(val_pred, y_val.to(device)).item()

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | train={epoch_loss:.4f} | val={val_loss:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "stats": stats,
        "history": history,
    }, MODEL_PATH)
    print(f"✅ 模型已保存：{MODEL_PATH}")

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val.to(device)).cpu().numpy()
    val_true = y_val.numpy()

    y_mean, y_std = stats["y_mean"], stats["y_std"]
    val_pred_denorm = val_pred * y_std + y_mean
    val_true_denorm = val_true * y_std + y_mean

    from sklearn.metrics import mean_absolute_error, r2_score
    mae_CL = mean_absolute_error(val_true_denorm[:, 0], val_pred_denorm[:, 0])
    mae_CD = mean_absolute_error(val_true_denorm[:, 1], val_pred_denorm[:, 1])
    r2_CL = r2_score(val_true_denorm[:, 0], val_pred_denorm[:, 0])
    r2_CD = r2_score(val_true_denorm[:, 1], val_pred_denorm[:, 1])

    print(f"\n📊 验证集性能：")
    print(f"  CL: MAE = {mae_CL:.4f}, R² = {r2_CL:.4f}")
    print(f"  CD: MAE = {mae_CD:.6f}, R² = {r2_CD:.4f}")

    return history, stats


if __name__ == "__main__":
    train(n_epochs=100)