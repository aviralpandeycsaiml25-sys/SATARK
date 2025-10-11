import pandas as pd, random, datetime

cities = [
    ("Delhi", 28.6139, 77.2090),
    ("Mumbai", 19.0760, 72.8777),
    ("Chennai", 13.0827, 80.2707),
    ("Kolkata", 22.5726, 88.3639),
    ("Bengaluru", 12.9716, 77.5946),
    ("Hyderabad", 17.3850, 78.4867),
    ("Ahmedabad", 23.0225, 72.5714),
    ("Pune", 18.5204, 73.8567),
    ("Jaipur", 26.9124, 75.7873),
    ("Lucknow", 26.8467, 80.9462),
]

crime_types = ["Theft", "Assault", "Burglary", "Robbery", "Fraud", "Vandalism", "Kidnapping"]
weights = [0.35, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05]

events = []
start, end = datetime.datetime(2018, 1, 1), datetime.datetime(2024, 12, 31)

for _ in range(2000):
    city, base_lat, base_lon = random.choice(cities)
    lat = base_lat + random.uniform(-0.25, 0.25)
    lon = base_lon + random.uniform(-0.25, 0.25)
    dt = start + (end - start) * random.random()
    category = random.choices(crime_types, weights)[0]
    events.append([city, lat, lon, dt.isoformat(), category])

df = pd.DataFrame(events, columns=["city", "latitude", "longitude", "timestamp", "category"])
df.to_csv("crimes.csv", index=False)
print(f"✅ crimes.csv generated with {len(df)} records.")
