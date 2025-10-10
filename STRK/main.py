from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
from sklearn.cluster import KMeans

app = FastAPI(title="SATARK - Crime Prediction & Analysis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")

df = pd.read_csv("crimes.csv", parse_dates=["timestamp"])
by_day = df['timestamp'].dt.day_name().value_counts().to_dict()
by_hour = df['timestamp'].dt.hour.value_counts().to_dict()

@app.get("/api/crimes")
def get_crimes():
    features = []
    for _, row in df.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row.longitude, row.latitude]},
            "properties": {
                "city": row.city,
                "category": row.category,
                "timestamp": row.timestamp.isoformat()
            }
        })
    return {"type": "FeatureCollection", "features": features}

@app.get("/api/analysis")
def get_analysis():
    return {"by_day": by_day, "by_hour": by_hour}

@app.get("/api/predict")
def get_predict():
    peak_day = max(by_day, key=by_day.get)
    peak_hour = max(by_hour, key=by_hour.get)
    kmeans = KMeans(n_clusters=7, random_state=0).fit(df[['latitude','longitude']])
    centers = kmeans.cluster_centers_.tolist()
    labels = kmeans.labels_
    counts = [int((labels==i).sum()) for i in range(7)]
    hotspots = [{"latitude": c[0], "longitude": c[1], "count": cnt} for c, cnt in zip(centers, counts)]
    return {"peak_day": peak_day, "peak_hour": int(peak_hour), "hotspots": hotspots}
