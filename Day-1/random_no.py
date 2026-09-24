import numpy as np

numbers = np.random.randint(1, 101, 10)

print("Random numbers:")
print(numbers)

print("Even numbers:")

for num in numbers:
    if num % 2 == 0:
        print(num)