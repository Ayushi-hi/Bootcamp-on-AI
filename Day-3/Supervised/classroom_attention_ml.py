import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# =========================================================
# 1. LOAD DATASET
# =========================================================

df = pd.read_csv("classroom_attention.csv")

print("========== CLASSROOM ATTENTION DATA ==========")

print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()


# =========================================================
# 2. CHECK MISSING VALUES
# =========================================================

print("\n========== MISSING VALUES ==========")

print(df.isnull().sum())


# =========================================================
# 3. ATTENTION DISTRIBUTION
# =========================================================

print("\n========== ATTENTION COUNTS ==========")

print(df["Attention"].value_counts())


# =========================================================
# 4. VISUALIZATION
# =========================================================

plt.figure(figsize=(7, 5))

df["Attention"].value_counts().plot(kind="bar")

plt.title("Classroom Attention Distribution")
plt.xlabel("Attention Level")
plt.ylabel("Number of Students")

plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# =========================================================
# 5. SELECT FEATURES
# =========================================================

features = [
    "Study_Hours",
    "Sleep_Hours",
    "Attendance",
    "Notes",
    "Phone_Usage",
    "Participation"
]

X = df[features]

y = df["Attention"]


# =========================================================
# 6. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# =========================================================
# 7. CREATE MODEL
# =========================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# =========================================================
# 8. TRAIN MODEL
# =========================================================

model.fit(X_train, y_train)

print("\nModel training completed!")


# =========================================================
# 9. PREDICTION
# =========================================================

y_pred = model.predict(X_test)


# =========================================================
# 10. ACCURACY
# =========================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL RESULT ==========")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

# 11. CLASSIFICATION REPORT

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# =========================================================
# 12. CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot()

plt.title("Classroom Attention - Confusion Matrix")

plt.tight_layout()
plt.show()


# =========================================================
# 13. FEATURE IMPORTANCE
# =========================================================

importance = model.feature_importances_

plt.figure(figsize=(8, 5))

plt.bar(
    features,
    importance
)

plt.title("Feature Importance")

plt.xlabel("Features")

plt.ylabel("Importance")

plt.xticks(rotation=30)

plt.tight_layout()
plt.show()


# =========================================================
# 14. ACTUAL VS PREDICTED
# =========================================================

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\n========== ACTUAL VS PREDICTED ==========")

print(comparison)


# =========================================================
# 15. PREDICT A NEW STUDENT
# =========================================================

new_student = pd.DataFrame({

    "Study_Hours": [4],

    "Sleep_Hours": [7],

    "Attendance": [88],

    "Notes": [8],

    "Phone_Usage": [2],

    "Participation": [8]
})


prediction = model.predict(
    new_student
)


print("\n========== NEW STUDENT ==========")

print(
    "Predicted Attention:",
    prediction[0]
)


# =========================================================
# 16. PREDICTION PROBABILITY
# =========================================================

probability = model.predict_proba(
    new_student
)

print("\nPrediction Probabilities:")

for class_name, prob in zip(
    model.classes_,
    probability[0]
):

    print(
        class_name,
        ":",
        round(prob * 100, 2),
        "%"
    )