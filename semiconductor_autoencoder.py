import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SensorAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, bottleneck_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


def fit_autoencoder(
    train_loader,
    val_loader,
    input_dim,
    lr=1e-3,
    epochs=50,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SensorAutoEncoder(input_dim=input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
            X_hat = model(X_batch)
            loss = criterion(X_hat, X_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                X_hat = model(X_batch)
                loss = criterion(X_hat, X_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_loss


def compute_reconstruction_scores(model, loader, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)

    mse_list = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            X_hat = model(X_batch)
            mse = torch.mean((X_hat - X_batch) ** 2, dim=1)
            mse_list.append(mse.cpu().numpy())

    mse_array = np.concatenate(mse_list, axis=0)
    return mse_array


def load_sensor_dataset(path_or_df, target_columns=None):
    if isinstance(path_or_df, str):
        df = pd.read_csv(path_or_df)
    elif isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        raise TypeError("path_or_df must be str or pandas.DataFrame")

    if target_columns is not None:
        df = df[target_columns]

    if df.isna().any().any():
        df = df.fillna(method="ffill").fillna(method="bfill").fillna(0)

    return df


def make_dataloaders(df, test_size=0.2, val_size=0.2, batch_size=128, random_state=42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values.astype(np.float32))

    X_train_val, X_test = train_test_split(
        X_scaled, test_size=test_size, random_state=random_state, shuffle=True
    )
    X_train, X_val = train_test_split(
        X_train_val, test_size=val_size, random_state=random_state, shuffle=True
    )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.zeros(len(X_train))),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.zeros(len(X_val))),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.zeros(len(X_test))),
        batch_size=batch_size,
        shuffle=False,
    )

    return scaler, train_loader, val_loader, test_loader


def determine_threshold(scores, percentile=99):
    return np.percentile(scores, percentile)


def infer_anomalies(model, dataloader, threshold, device=None):
    scores = compute_reconstruction_scores(model, dataloader, device)
    anomalies = scores > threshold
    return scores, anomalies


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path, input_dim, hidden_dim=64, bottleneck_dim=16, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SensorAutoEncoder(input_dim=input_dim, hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # CSV 파일 기반 학습
    csv_path = "sensor_normal_1000.csv"  # 현재 워크스페이스에 존재하는 파일 경로
    print(f"Load CSV: {csv_path}")
    df = load_sensor_dataset(csv_path, target_columns=["temp", "pressure", "vibration", "voltage"])

    scaler, train_loader, val_loader, test_loader = make_dataloaders(df, test_size=0.2, val_size=0.1, batch_size=128)

    print("Training autoencoder...")
    model, _ = fit_autoencoder(train_loader, val_loader, input_dim=df.shape[1], lr=1e-3, epochs=50)

    print("평가: 정상 데이터 재구성 오류 계산")
    test_scores = compute_reconstruction_scores(model, test_loader)
    threshold = determine_threshold(test_scores, percentile=99)
    print(f"선택된 anomaly threshold (99percentile): {threshold:.6f}")

    save_path = "models/semiconductor_autoencoder.pth"
    save_model(model, save_path)
    print(f"학습 완료. 모델 저장: {save_path}")

    # 이상 샘플이 있는 경우 다음처럼 테스트할 수 있습니다.
    # anom_df = load_sensor_dataset("sensor_normal_with_anomaly_1000.csv", target_columns=["temp","pressure","vibration","voltage"])
    # X_anom = scaler.transform(anom_df.values.astype(np.float32))
    # anomaly_loader = DataLoader(TensorDataset(torch.from_numpy(X_anom), torch.zeros(len(X_anom))), batch_size=128)
    # anom_scores, anom_flags = infer_anomalies(model, anomaly_loader, threshold)
    # print(f"이상샘플 중 감지된 비율: {anom_flags.mean()*100:.2f}% ({anom_flags.sum()}/{len(anom_flags)})")

    # 이상 샘플이 있는 경우
    anom_df = load_sensor_dataset("sensor_normal_with_anomaly_1000.csv", target_columns=["temp","pressure","vibration","voltage"])
    X_anom = scaler.transform(anom_df.values.astype(np.float32))
    anomaly_loader = DataLoader(TensorDataset(torch.from_numpy(X_anom), torch.zeros(len(X_anom))), batch_size=128)
    anom_scores, anom_flags = infer_anomalies(model, anomaly_loader, threshold)

    # 여기 바로 Precision/Recall 계산
    labels = pd.read_csv("sensor_normal_with_anomaly_1000.csv")["label"].values
    preds = anom_flags

    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    print("Precision:", precision)
    print("Recall:", recall)

    # 기존 출력
    print(f"이상샘플 중 감지된 비율: {anom_flags.mean()*100:.2f}% ({anom_flags.sum()}/{len(anom_flags)})")