import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix, classification_report
from semiconductor_autoencoder import (
    load_sensor_dataset,
    make_dataloaders,
    load_model,
    compute_reconstruction_scores,
    determine_threshold,
)


def run_evaluation():
    model = load_model("models/semiconductor_autoencoder.pth", input_dim=4)

    normal_df = load_sensor_dataset("sensor_normal_1000.csv", ["temp", "pressure", "vibration", "voltage"])
    _, _, _, test_loader = make_dataloaders(normal_df, test_size=0.2, val_size=0.1, batch_size=128)
    normal_scores = compute_reconstruction_scores(model, test_loader)

    for percentile in [95, 97, 99, 99.5]:
        th = determine_threshold(normal_scores, percentile)
        print(f"threshold {percentile}%: {th:.6f}")

    threshold = determine_threshold(normal_scores, 99)

    anom_df = pd.read_csv("sensor_normal_with_anomaly_1000.csv")
    X_all = anom_df[["temp", "pressure", "vibration", "voltage"]].values.astype(np.float32)
    y_true = anom_df["label"].values

    anomaly_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X_all), torch.zeros(len(X_all))),
        batch_size=128,
        shuffle=False,
    )

    all_scores = compute_reconstruction_scores(model, anomaly_loader)
    y_pred = (all_scores > threshold).astype(int)

    print("\n==== Evaluation (threshold 99 percentile) ====")
    print("AUC:", roc_auc_score(y_true, all_scores))
    print("F1:", f1_score(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification report:\n", classification_report(y_true, y_pred))


if __name__ == "__main__":
    run_evaluation()
