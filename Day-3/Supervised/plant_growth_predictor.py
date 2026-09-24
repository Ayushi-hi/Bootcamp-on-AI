import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# LOAD DATASET

df = pd.read_csv("plant_growth_data.csv")

print("========== PLANT GROWTH DATA ==========")
print(df.head())

print("\nDataset Shape:", df.shape)
print("\nDataset Information:")
print(df.info())


# BASIC DATA ANALYSIS

print("\nGrowth Level Counts:")
print(df["Growth_Level"].value_counts())


# VISUALIZATION - GROWTH LEVEL DISTRIBUTION

plt.figure(figsize=(7, 5))

df["Growth_Level"].value_counts().plot(kind="bar")

plt.title("Plant Growth Level Distribution")
plt.xlabel("Growth Level")
plt.ylabel("Number of Plants")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# VISUALIZATION - WATER VS GROWTH

plt.figure(figsize=(8, 5))

for growth in df["Growth_Level"].unique():

    data = df[df["Growth_Level"] == growth]

    plt.scatter(
        data["Water_Amount"],
        data["Soil_Moisture"],
        label=growth
    )

plt.title("Water Amount vs Soil Moisture")
plt.xlabel("Water Amount (ml)")
plt.ylabel("Soil Moisture")
plt.legend()

plt.tight_layout()
plt.show()


# SELECT FEATURES

features = [
    "Water_Amount",
    "Sunlight_Hours",
    "Temperature",
    "Soil_Moisture",
    "Fertilizer_Amount"
]

X = df[features]

y = df["Growth_Level"]


# SPLIT DATA

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# CREATE ML MODEL

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# TRAIN MODEL

model.fit(X_train, y_train)

print("\nModel training completed!")

# MAKE PREDICTIONS

y_pred = model.predict(X_test)


# MODEL ACCURACY

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL RESULT ==========")
print("Accuracy:", round(accuracy * 100, 2), "%")

# CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot()

plt.title("Plant Growth Prediction - Confusion Matrix")
plt.tight_layout()
plt.show()

# FEATURE IMPORTANCE

importance = model.feature_importances_

plt.figure(figsize=(8, 5))

plt.bar(features, importance)

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.xticks(rotation=30)

plt.tight_layout()
plt.show()

# ACTUAL VS PREDICTED 

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\n========== ACTUAL VS PREDICTED ==========")
print(comparison)


# PREDICT A NEW PLANT

new_plant = pd.DataFrame({
    "Water_Amount": [500],
    "Sunlight_Hours": [7],
    "Temperature": [25],
    "Soil_Moisture": [60],
    "Fertilizer_Amount": [20]
})

prediction = model.predict(new_plant)

print("\n========== NEW PLANT PREDICTION ==========")

print(
    "Predicted Growth Level:",
    prediction[0]
)


# PREDICTION PROBABILITY
probability = model.predict_proba(new_plant)

print("\nPrediction Probabilities:")

for class_name, prob in zip(model.classes_, probability[0]):

    print(
        class_name,
        ":", round(prob * 100, 2), "%"
    )