# app.py
import os
import json
import math
import pandas as pd
from flask import Flask, render_template, request
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

app = Flask(__name__)

SCALER_PATH = "models/scaler.pkl"
SUMMARY_PATH = "models/cluster_summary.json"
DATA_PATH = "data/wifi_readings.csv"
ALT_LABELED = "models/wifi_labeled.csv"
PLOTS_DIR = "static/plots"

def load_resources():
    scaler = None
    cluster_summary = []
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
        except Exception:
            scaler = None
    if os.path.exists(SUMMARY_PATH):
        try:
            with open(SUMMARY_PATH, "r") as f:
                cluster_summary = json.load(f)
        except Exception:
            cluster_summary = []
    return scaler, cluster_summary

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

@app.route("/", methods=["GET", "POST"])
def index():
    scaler, cluster_summary = load_resources()
    if request.method == "POST":
        try:
            lat = float(request.form.get("lat"))
            lon = float(request.form.get("lon"))
            rssi = float(request.form.get("rssi_dbm"))
            devices = int(request.form.get("device_count"))
        except Exception as e:
            return f"Invalid input: {e}", 400

        assigned = {"cluster": -1, "distance_m": None, "note": "No hotspot detected (noise)"}
        if not cluster_summary:
            # cluster info missing — instruct to run train.py
            assigned["note"] = "Cluster summaries not found. Please run train.py to generate hotspots."
            return render_template("result.html", assigned=assigned, plot_url=None)

        # find nearest cluster center (meters)
        best = None
        best_d = float("inf")
        for c in cluster_summary:
            d = haversine(lat, lon, c["center_lat"], c["center_lon"])
            if d < best_d:
                best_d = d
                best = c

        if best and best_d <= 150:
            assigned = {
                "cluster": int(best["cluster"]),
                "distance_m": round(best_d, 1),
                "note": f"Belongs to hotspot {best['cluster']} (approx {best['count']} readings)",
            }
        else:
            assigned = {
                "cluster": -1,
                "distance_m": round(best_d, 1) if best else None,
                "note": "No hotspot detected (noise / outlier)",
            }

        # Load labeled dataset for plotting
        df = None
        for path in [DATA_PATH, ALT_LABELED]:
            if os.path.exists(path):
                try:
                    tmp = pd.read_csv(path)
                    if "cluster" in tmp.columns:
                        df = tmp
                        break
                except Exception:
                    continue

        if df is None:
            # No labeled dataset, can't plot; return result without plot
            return render_template("result.html", assigned=assigned, plot_url=None)

        # Plot: clusters, noise, centers, and user point
        plt.figure(figsize=(7, 7))
        # clusters
        labs = sorted([l for l in df["cluster"].unique() if l != -1])
        for lab in labs:
            mask = df["cluster"] == lab
            plt.scatter(df.loc[mask, "lon"], df.loc[mask, "lat"], s=10, alpha=0.6, label=f"Cluster {lab}")
        # noise
        noise_mask = df["cluster"] == -1
        if noise_mask.any():
            plt.scatter(df.loc[noise_mask, "lon"], df.loc[noise_mask, "lat"], s=6, alpha=0.4, color="grey", label="Noise")
        # centers
        for c in cluster_summary:
            plt.scatter(c["center_lon"], c["center_lat"], marker="*", s=160, c="yellow", edgecolors="k")
        # user
        plt.scatter(lon, lat, s=180, c="red", marker="X", edgecolors="k", linewidths=1.5, label="You")

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Wi-Fi Hotspot Map (your location highlighted)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plot_path = os.path.join(PLOTS_DIR, "user_map.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        return render_template("result.html", assigned=assigned, plot_url=plot_path)
    else:
        ranges = {"lat_min": 37.74, "lat_max": 37.80, "lon_min": -122.445, "lon_max": -122.395, "rssi_min": -100, "rssi_max": -20}
        return render_template("index.html", ranges=ranges)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5400, debug=True)
