# train.py
import os
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "data"
MODELS_DIR = "models"
PLOTS_DIR = "static/plots"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

RAW_PATH = os.path.join(DATA_DIR, "wifi_readings_raw.csv")
DATA_PATH = os.path.join(DATA_DIR, "wifi_readings.csv")          # final labeled file
LABELED_FALLBACK = os.path.join(MODELS_DIR, "wifi_labeled.csv")  # also saved here

def atomic_write(df: pd.DataFrame, target_path: str):
    """Write dataframe to CSV atomically; fall back to timestamped file if replace fails."""
    tmp = target_path + ".tmp"
    df.to_csv(tmp, index=False)
    try:
        os.replace(tmp, target_path)
    except PermissionError as e:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = target_path.replace(".csv", f"_labeled_{ts}.csv")
        print(f"[WARNING] Could not replace {target_path} (permission). Writing fallback {fallback}")
        os.replace(tmp, fallback)
        return fallback
    return target_path

def generate_synthetic():
    """Generate synthetic Wi-Fi reading dataset (same approach as earlier)."""
    np.random.seed(42)
    hotspot_centers = [
        (37.775, -122.418, -40),
        (37.788, -122.405, -45),
        (37.765, -122.435, -50)
    ]

    points = []
    for (clat, clon, crssi) in hotspot_centers:
        k = 300
        lats = np.random.normal(clat, 0.002, k)
        lons = np.random.normal(clon, 0.002, k)
        rssi = np.random.normal(crssi, 4, k)
        for lat, lon, r in zip(lats, lons, rssi):
            points.append([lat, lon, r, np.random.randint(1, 8)])

    m = 300
    lats = np.random.uniform(37.74, 37.80, m)
    lons = np.random.uniform(-122.445, -122.395, m)
    rssi = np.random.normal(-80, 8, m)
    for lat, lon, r in zip(lats, lons, rssi):
        points.append([lat, lon, r, np.random.randint(0, 4)])

    df = pd.DataFrame(points, columns=["lat", "lon", "rssi_dbm", "device_count"])
    return df

def compute_and_save_clusters(df: pd.DataFrame, eps=0.35, min_samples=15):
    """Run DBSCAN on df (lat,lon,rssi_dbm,device_count), add 'cluster', save artifacts & plot."""
    # Clean input
    df = df.copy()
    for col in ["lat", "lon", "rssi_dbm", "device_count"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df = df.dropna(subset=["lat", "lon", "rssi_dbm", "device_count"])

    X = df[["lat", "lon", "rssi_dbm", "device_count"]].astype(float).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = db.fit_predict(Xs)

    df["cluster"] = labels.astype(int)

    # save scaler
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    # compute cluster summary
    cluster_summary = []
    unique_labels = sorted([l for l in set(labels) if l != -1])
    for lab in unique_labels:
        subset = df[df["cluster"] == lab]
        center = subset[["lat", "lon"]].mean().to_list()
        avg_rssi = float(subset["rssi_dbm"].mean())
        avg_devices = float(subset["device_count"].mean())
        cluster_summary.append({
            "cluster": int(lab),
            "center_lat": float(center[0]),
            "center_lon": float(center[1]),
            "avg_rssi": avg_rssi,
            "avg_devices": avg_devices,
            "count": int(len(subset))
        })

    with open(os.path.join(MODELS_DIR, "cluster_summary.json"), "w") as f:
        json.dump(cluster_summary, f, indent=2)

    # save labeled data (atomic)
    out_path = atomic_write(df, DATA_PATH)
    # also save a copy in models
    df.to_csv(LABELED_FALLBACK, index=False)

    # plot
    plt.figure(figsize=(8, 8))
    for lab in unique_labels:
        mask = df["cluster"] == lab
        plt.scatter(df.loc[mask, "lon"], df.loc[mask, "lat"], s=12, alpha=0.7, label=f"Cluster {lab}")
    noise_mask = df["cluster"] == -1
    if noise_mask.any():
        plt.scatter(df.loc[noise_mask, "lon"], df.loc[noise_mask, "lat"], s=8, alpha=0.5, color="grey", label="Noise")

    for c in cluster_summary:
        plt.scatter(c["center_lon"], c["center_lat"], marker="*", s=200, edgecolors="k", linewidths=1.2, c="yellow")
        plt.text(c["center_lon"] + 0.0004, c["center_lat"] + 0.0004, f"Hotspot {c['cluster']}", fontsize=9, weight="bold")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Wi-Fi Hotspot Clusters (DBSCAN)")
    plt.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cluster_map.png"), dpi=150)
    plt.close()

    return df, cluster_summary, out_path

if __name__ == "__main__":
    # If user provided data/wifi_readings.csv, prefer it
    if os.path.exists(DATA_PATH):
        print(f"Found existing {DATA_PATH} — loading.")
        df_in = pd.read_csv(DATA_PATH)
        # if it already has 'cluster', we only summarize & plot
        if "cluster" in df_in.columns:
            print("Existing dataset already contains 'cluster' column — building summaries and plots.")
            df, summary, saved = compute_and_save_clusters(df_in)  # compute_and_save_clusters will overwrite to ensure standardized pipeline
        else:
            print("'cluster' column not found — running DBSCAN on existing readings.")
            df, summary, saved = compute_and_save_clusters(df_in)
    else:
        # check for raw copy saved earlier
        if os.path.exists(RAW_PATH):
            print(f"Found raw file {RAW_PATH} — using it.")
            df_raw = pd.read_csv(RAW_PATH)
            df, summary, saved = compute_and_save_clusters(df_raw)
        else:
            print("No input data found. Generating synthetic dataset and running DBSCAN.")
            df_gen = generate_synthetic()
            # save raw copy so user can inspect
            df_gen.to_csv(RAW_PATH, index=False)
            df, summary, saved = compute_and_save_clusters(df_gen)

    print("Generated data, ran DBSCAN, saved artifacts:")
    print(f" - labeled readings: {saved}")
    print(f" - models/wifi_labeled.csv: {LABELED_FALLBACK}")
    print(f" - models/cluster_summary.json: {os.path.join(MODELS_DIR,'cluster_summary.json')}")
    print(f" - plot: {os.path.join(PLOTS_DIR,'cluster_map.png')}")
