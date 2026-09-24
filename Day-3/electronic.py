import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


mydata = {
    "voltage": [
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12
    ],

    "current": [
        48, 101, 149, 202, 251,
        301, 349, 401, 451, 498,
        551, 602
    ]
}

df = pd.DataFrame(mydata)

print("Electronic Circuit Dataset:")
print(df)


df.to_csv("voltage_current.csv", index=False)

print("Data saved to voltage_current.csv")


plt.figure(figsize=(8, 5))

plt.scatter(
    df["voltage"],
    df["current"]
)

plt.xlabel("Voltage (V)")
plt.ylabel("Current (mA)")
plt.title("Voltage vs Current")

plt.grid(True)
plt.show()


X = df[["voltage"]]
Y = df["current"]

print("X shape:", X.shape)
print("Y shape:", Y.shape)


X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))



model = LinearRegression()

model.fit(X_train, Y_train)

print("Linear Regression model trained!")


Y_pred = model.predict(X_test)

print("Actual Current:")
print(Y_test.values)

print("Predicted Current:")
print(Y_pred)

slope = model.coef_[0]
intercept = model.intercept_

print("Slope:", slope)
print("Intercept:", intercept)

print(
    f"Equation: Current = {slope:.2f} × Voltage + {intercept:.2f}"
)


mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("Model Evaluation:")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

new_voltage = pd.DataFrame({
    "voltage": [7.5]
})

predicted_current = model.predict(new_voltage)

print(
    "Predicted current for 7.5 V:",
    predicted_current[0],
    "mA"
)


plt.figure(figsize=(8, 5))

plt.scatter(
    X,
    Y,
    label="Actual Data"
)

plt.plot(
    X,
    model.predict(X),
    linewidth=2,
    label="Regression Line"
)

plt.xlabel("Voltage (V)")
plt.ylabel("Current (mA)")
plt.title("Linear Regression - Voltage vs Current")

plt.legend()
plt.grid(True)

plt.show()