from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from semiconductor_autoencoder import load_model, load_sensor_dataset, make_dataloaders, compute_reconstruction_scores

app = FastAPI()

class SensorPoint(BaseModel):
    temp: float
    pressure: float
    vibration: float
    voltage: float

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "semiconductor_autoencoder.pth"
SENSOR_CSV = BASE_DIR / "sensor_normal_1000.csv"

model = None
scaler = None
threshold = None
startup_ok = False


def initialize_model():
    global model, scaler, threshold, startup_ok
    try:
        model = load_model(str(MODEL_PATH), input_dim=4)
        norm_df = load_sensor_dataset(str(SENSOR_CSV), ["temp", "pressure", "vibration", "voltage"])

        scaler = StandardScaler()
        scaler.fit(norm_df.values.astype(np.float32))

        _, _, _, test_loader = make_dataloaders(norm_df, test_size=0.2, val_size=0.1, batch_size=128)
        scores = compute_reconstruction_scores(model, test_loader)
        threshold = float(np.percentile(scores, 99))

        startup_ok = True
        return {
            "status": "ok",
            "threshold": threshold,
            "model_path": str(MODEL_PATH),
            "data_path": str(SENSOR_CSV),
        }
    except Exception as e:
        startup_ok = False
        raise RuntimeError(f"Startup initialization failed: {e}")


@app.on_event("startup")
def startup_event():
    initialize_model()

@app.get("/health")
def health():
    return {
        "status": "ok" if startup_ok else "error",
        "model_loaded": model is not None,
        "threshold": float(threshold) if threshold is not None else None,
    }


@app.post("/infer")
def infer(sensor: SensorPoint):
    if not startup_ok or model is None or scaler is None or threshold is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    x = np.array([[sensor.temp, sensor.pressure, sensor.vibration, sensor.voltage]], dtype=np.float32)
    try:
        x_scaled = scaler.transform(x)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"input scaling failed: {e}")

    x_tensor = torch.from_numpy(x_scaled)
    with torch.no_grad():
        x_hat = model(x_tensor)
        score = float(torch.mean((x_hat - x_tensor) ** 2).cpu().numpy())
    is_anomaly = score > threshold

    return {
        "reconstruction_score": score,
        "threshold": float(threshold),
        "is_anomaly": bool(is_anomaly),
    }


if __name__ == "__main__":
    import uvicorn

    try:
        initialize_model()
        uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")
    except Exception as e:
        print(f"Startup failed: {e}")
        raise


