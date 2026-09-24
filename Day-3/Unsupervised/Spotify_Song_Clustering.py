import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Check environment
# --------------------------------------------------

print("Environment ready!")
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("scikit-learn:", __import__("sklearn").__version__)


# --------------------------------------------------
# 2. Create Spotify-style dataset
# --------------------------------------------------

mydata = {
    "song": [
        "Blinding Lights", "Levitating", "Stay", "Dance Monkey",
        "Shape of You", "Don't Start Now", "Industry Baby",
        "Perfect", "Photograph", "Someone Like You",
        "Lovely", "Arcade", "All of Me", "Let Her Go",
        "Believer", "Thunder", "Enemy", "Centuries",
        "Sunflower", "Heat Waves",
        "Peaches", "As It Was", "Watermelon Sugar",
        "Bad Guy", "Havana", "Senorita", "Closer",
        "Counting Stars", "Faded", "Cheap Thrills"
    ],

    "energy": [
        80, 85, 82, 78, 75, 84, 88, 40, 35, 30,
        28, 25, 32, 38, 90, 87, 92, 85, 65, 70,
        76, 72, 74, 73, 68, 65, 82, 80, 78, 86
    ],

    "danceability": [
        85, 90, 82, 88, 92, 86, 84, 55, 50, 45,
        40, 42, 48, 52, 65, 70, 75, 68, 78, 80,
        83, 84, 82, 89, 85, 82, 88, 75, 72, 90
    ],

    "tempo": [
        86, 103, 170, 98, 96, 124, 150, 64, 72, 68,
        76, 72, 65, 78, 125, 168, 155, 132, 90, 81,
        90, 174, 97, 135, 105, 117, 96, 122, 90, 112
    ],

    "acousticness": [
        10, 8, 12, 15, 20, 7, 5, 80, 85, 90,
        88, 92, 87, 82, 10, 8, 5, 12, 25, 20,
        18, 15, 12, 10, 22, 18, 10, 20, 15, 8
    ],

    "valence": [
        75, 92, 80, 85, 90, 88, 76, 65, 55, 30,
        28, 25, 35, 45, 70, 82, 78, 65, 85, 88,
        80, 86, 90, 60, 82, 80, 72, 78, 62, 91
    ]
}


df = pd.DataFrame(mydata)


# --------------------------------------------------
# 3. Display dataset
# --------------------------------------------------

print("\nSpotify Song Dataset:")
print(df)

print("\nDataset shape:", df.shape)


# --------------------------------------------------
# 4. Save dataset
# --------------------------------------------------

df.to_csv("spotify_songs.csv", index=False)

print("\nData saved to spotify_songs.csv")


# --------------------------------------------------
# 5. Select features for clustering
# --------------------------------------------------

features = [
    "energy",
    "danceability",
    "tempo",
    "acousticness",
    "valence"
]

X = df[features]

print("\nFeatures used for clustering:")
print(features)


# --------------------------------------------------
# 6. Standardize the data
# --------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# --------------------------------------------------
# 7. Create K-Means model
# --------------------------------------------------

model = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)


# --------------------------------------------------
# 8. Train the model
# --------------------------------------------------

model.fit(X_scaled)

print("\nK-Means model trained!")


# --------------------------------------------------
# 9. Add cluster labels
# --------------------------------------------------

df["cluster"] = model.labels_

print("\nSongs with cluster numbers:")
print(
    df[
        ["song", "energy", "danceability",
         "tempo", "acousticness", "valence", "cluster"]
    ]
)


# --------------------------------------------------
# 10. Count songs in each cluster
# --------------------------------------------------

print("\nNumber of songs in each cluster:")

print(
    df["cluster"]
    .value_counts()
    .sort_index()
)


# --------------------------------------------------
# 11. Display cluster centers
# --------------------------------------------------

cluster_centers = scaler.inverse_transform(
    model.cluster_centers_
)

cluster_centers_df = pd.DataFrame(
    cluster_centers,
    columns=features
)

print("\nCluster Centers:")
print(cluster_centers_df)


# --------------------------------------------------
# 12. Visualize clusters
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    df["energy"],
    df["danceability"],
    c=df["cluster"],
    s=100
)

plt.xlabel("Energy")
plt.ylabel("Danceability")
plt.title("Spotify Song Clustering using K-Means")

plt.grid(True)

plt.show()