import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Environment ready!")
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("scikit-learn:", __import__("sklearn").__version__)

mydata = {
    "study_hours":[
        1,2,3,4,5,6,7,8,9,10,
        11,12,13,14,15,16,17,18,19,20,
        21,22,23,24,25,26,27,28,29,30
    ],
    "attendance":[40,44,40,45,90,92,95,96,97,98,
        45,56,20,35,49,86,76,56,100,100,
        54,50,32,25,68,32,85,68,32,85
    ],
    "assignment":[37,40,29,39,90,100,65,75,85,95,
        55,24,15,46,95,100,60,70,80,90,
        50,55,43,23,90,100,65,75,85,95
    ],
    "result":[0,0,0,0,1,1,1,1,1,1,
        0,0,0,0,1,1,1,1,1,1,
        0,0,0,0,1,1,1,1,1,1
    ]

}
df = pd.DataFrame(mydata)
print("Student DataSet:")
print(df)
df.to_csv("student.csv", index=False)
print("Data saved to student.csv")

print("Average performance by result:")
print(df.groupby("result")[
    ["study_hours", "attendance", "assignment"]].mean())

plt.figure(figsize=(10, 6))

plt.scatter(
    df["study_hours"], df["assignment"],
    c = df["result"]
)

plt.xlabel("Study Hours")
plt.ylabel("Assignment Score")
plt.title("Student Performance Pattern")

plt.show()

X = df[["study_hours", "attendance", "assignment"]]
Y = df["result"]
print("X shape:", X.shape)
print("Y shape:", Y.shape)


X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42,
    stratify=Y
)
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

depths = [3, 4, 5]
accuracies = []

for depth in depths:

    print("Decision Tree - max_depth =", depth)
    

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    acc = accuracy_score(Y_test, Y_pred)

    accuracies.append(acc)

    print("Accuracy:", acc)

    print("Confusion Matrix:")
    print(confusion_matrix(Y_test, Y_pred))

    print("Classification Report:")
    print(
        classification_report(
            Y_test,
            Y_pred,
            zero_division=0
        )
    )

comparison_df = pd.DataFrame({
    "Max Depth": depths,
    "Accuracy": accuracies
})

print("COMPARATIVE ANALYSIS")
print(comparison_df)


plt.figure(figsize=(7, 4))

plt.plot(
    depths,
    accuracies,
    marker="o"
)

plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree: Effect of Max Depth")

plt.xticks(depths)
plt.grid(True)

plt.show()

final_model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

final_model.fit(X_train, Y_train)

print("\nFinal model trained with max_depth = 3")


plt.figure(figsize=(13, 7))

plot_tree(
    final_model,
    feature_names=X.columns,
    class_names=["Fail", "Pass"],
    filled=True,
    rounded=True
)

plt.title("Decision Tree - max_depth = 3")

plt.show()