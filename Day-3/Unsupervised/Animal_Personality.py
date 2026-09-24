import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# --------------------------------------------------
# 1. Check environment
# --------------------------------------------------

print("Environment ready!")
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("scikit-learn:", __import__("sklearn").__version__)


# --------------------------------------------------
# 2. Create animal dataset
# --------------------------------------------------

mydata = {
    "animal": [
        "Lion", "Tiger", "Elephant", "Cheetah", "Wolf",
        "Fox", "Dog", "Cat", "Horse", "Rabbit",
        "Bear", "Monkey", "Dolphin", "Eagle", "Penguin",
        "Kangaroo", "Panda", "Giraffe", "Zebra", "Leopard"
    ],

    "strength": [
        95, 98, 100, 85, 88,
        55, 60, 50, 75, 30,
        95, 65, 70, 60, 45,
        70, 60, 65, 72, 90
    ],

    "speed": [
        80, 85, 100, 100, 75,
        70, 65, 60, 75, 45,
        55, 70, 65, 95, 50,
        80, 40, 55, 70, 95
    ],

    "agility": [
        75, 82, 95, 100, 85,
        92, 80, 90, 70, 85,
        60, 95, 80, 98, 45,
        88, 55, 50, 65, 92
    ],

    "intelligence": [
        75, 80, 85, 70, 90,
        88, 85, 82, 75, 70,
        78, 92, 90, 85, 65,
        72, 80, 60, 55, 82
    ],

    "social": [
        65, 55, 90, 40, 85,
        70, 95, 45, 80, 75,
        60, 95, 90, 50, 95,
        85, 80, 75, 70, 55
    ],

    "curiosity": [
        70, 65, 75, 60, 80,
        95, 90, 85, 70, 88,
        72, 100, 98, 75, 65,
        82, 85, 55, 60, 78
    ],

    "playfulness": [
        65, 60, 70, 55, 75,
        90, 100, 85, 80, 95,
        60, 100, 90, 70, 85,
        95, 75, 65, 70, 65
    ],

    "fearlessness": [
        95, 92, 90, 88, 82,
        65, 70, 55, 75, 40,
        85, 95, 80, 90, 45,
        78, 50, 60, 65, 88
    ]
}


df = pd.DataFrame(mydata)

print("\nAnimal Dataset:")
print(df)

print("\nDataset shape:")
print(df.shape)


# --------------------------------------------------
# 3. Save dataset
# --------------------------------------------------

df.to_csv("animal_personality.csv", index=False)

print("\nData saved to animal_personality.csv")


# --------------------------------------------------
# 4. Select features
# --------------------------------------------------

features = [
    "strength",
    "speed",
    "agility",
    "intelligence",
    "social",
    "curiosity",
    "playfulness",
    "fearlessness"
]

X = df[features]

print("\nNumber of original features:", X.shape[1])


# --------------------------------------------------
# 5. Standardize data
# --------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# --------------------------------------------------
# 6. Apply PCA
# --------------------------------------------------

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)


# --------------------------------------------------
# 7. Create reduced dataset
# --------------------------------------------------

pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

pca_df["animal"] = df["animal"]


print("\nReduced Dataset:")
print(pca_df)


# --------------------------------------------------
# 8. Explained variance
# --------------------------------------------------

print("\nExplained Variance Ratio:")

print(pca.explained_variance_ratio_)

total_variance = pca.explained_variance_ratio_.sum()

print(
    "\nTotal variance retained:",
    round(total_variance * 100, 2),
    "%"
)


# --------------------------------------------------
# 9. PCA component weights
# --------------------------------------------------

print("\nPCA Components:")

components = pd.DataFrame(
    pca.components_,
    columns=features,
    index=["PC1", "PC2"]
)

print(components)


# --------------------------------------------------
# 10. Plot Animal Personality Map
# --------------------------------------------------

plt.figure(figsize=(12, 8))

plt.scatter(
    pca_df["PC1"],
    pca_df["PC2"],
    s=120
)


# Add animal names
for i in range(len(pca_df)):

    plt.annotate(
        pca_df["animal"].iloc[i],
        (
            pca_df["PC1"].iloc[i],
            pca_df["PC2"].iloc[i]
        ),
        fontsize=9
    )


plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title(
    "Animal Personality Map using PCA"
)

plt.grid(True)

plt.tight_layout()

plt.show()