import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 100)

y = np.sin(x)

plt.plot(x, y)

plt.xlabel("X")
plt.ylabel("sin(X)")
plt.title("Sine Wave")

plt.grid()

plt.show()