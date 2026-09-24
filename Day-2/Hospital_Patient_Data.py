import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

mydata = {
    "PatientID": [
        "P001","P002","P003","P004","P005",
        "P006","P007","P008","P009","P010",
        "P011","P012","P013","P014","P015",
        "P016","P017","P018","P019","P020",
        "P021","P022","P023","P024","P025"
    ],

    "Patient_Name": [
        "AYUSH","RAJ","SURAJ","RAVI","KUMAR",
        "RAY","PAGLU","RAJESH","RAMESH","SURESH",
        "RAJIV","RANJEET","RAHUL","RAVI","RAJESH",
        "RAY","PAGLU","RAJESH","RAMESH","SURESH",
        "RAJIV","RANJEET","RAHUL","RAVI","RAJESH"
    ],

    "Age": [
        25,24,21,22,24,
        23,22,21,24,25,
        23,22,21,24,25,
        23,22,21,24,25,
        32,33,34,35,36
    ],

    "Gender": [
        "M","M","M","M","M",
        "M","M","M","M","M",
        "M","M","M","M","M",
        "M","M","M","M","M",
        "M","M","M","M","M"
    ],

    "Disease": [
        "Fever","Cold","Cough","Fever","Cold",
        "Cough","Fever","Cold","Cough","Fever",
        "Cold","Cough","Fever","Cold","Cough",
        "Cancer","Diabetes","Hypertension","Asthma","Allergy",
        "Tyhoid","Malaria","Dengue","Cholera","Hepatitis"
    ],

    "Date": [
    "2021-01-01","2021-01-02","2021-01-03",
    "2021-01-04","2021-01-05","2021-01-06",
    "2021-01-07","2021-01-08","2021-01-09",
    "2021-01-10","2021-01-11","2021-01-12",
    "2021-01-13","2021-01-14","2021-01-15",
    "2021-01-16","2021-01-17","2021-01-18",
    "2021-01-19","2021-01-20","2021-01-21",
    "2021-01-22","2021-01-23","2021-01-24",
    "2021-01-25"
],

    "Status": [
        "Recovered","Discharged","Recovered","Recovered","Critical",
        "Critical","Recovered","Recovered","Discharged","Recovered",
        "Recovered","Discharged","Recovered","Recovered","Critical",
        "Critical","Recovered","Recovered","Discharged","Recovered",
        "Recovered","Discharged","Recovered","Recovered","Critical"
    ],

    "Treatment-Cost": [
        1000,2000,1500,1200,2500,
        2200,1800,1600,2100,1900,
        1700,2300,1400,2400,2600,
        1300,1500,1700,1900,2100,
        2200,2300,2400,2500,2600
    ],
}


df = pd.DataFrame(mydata, columns=["PatientID", "Patient_Name", "Age", "Gender", "Disease", "Date", "Status", "Treatment-Cost"])

df.to_csv("Hospital_Patient_Data.csv", index=False)

print("Hospital patient data has been created successfully.")
print()


print("Environment ready!")
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", __import__("sklearn").__version__)
print()

print("First 5 patients:")
print(df.head())
print()


print("Average treatment cost by disease:")

print(
    df.groupby("Disease")["Treatment-Cost"].mean()
)

print()


print("Average age by disease:")

print(
    df.groupby("Disease")["Age"].mean()
)

print()


print("Patient status count:")

print(
    df["Status"].value_counts()
)

print()


le = LabelEncoder()
df["Disease_Code"] = le.fit_transform(df["Disease"])

df["Disease_Code"] = df["Disease"].map({
    "Cold": 0,
    "Cough": 1,
    "Fever": 2
})

# Status will be our TARGET
df["Status_Code"] = df["Status"].map({
    "Discharged": 0,
    "Recovered": 1,
    "Critical": 2
})


# Features
X = df[[
    "Age",
    "Disease_Code",
    "Treatment-Cost"
]]

# Target
y = df["Status_Code"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("Training data:", len(X_train))
print("Testing data:", len(X_test))
print()



model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)




prediction = model.predict(X_test)
print("Predictions:", prediction)
print("X", X_test)


accuracy = accuracy_score(y_test, prediction)
print(f"Accuracy:, {accuracy:.2%}")

plt.figure(figsize=(13,7))
plot_tree(
    model,
    feature_names=[
         "Age",
         "Disease_Code",
         "Treatment-Cost"],
         class_names=[
             "Discharged",
             "Recovered",
             "Critical"],
             filled=True
             )
plt.title("Hospital Patient Status Decision Tree")
plt.show()
