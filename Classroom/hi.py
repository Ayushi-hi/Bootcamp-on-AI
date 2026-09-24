import numpy as np

a = np.random.randint(0, 100, (6,8,7))
np.random.seed(33)

print("Shape:", a.shape)
print("Size:", a.size)
print("Dimension:", a.ndim)

print("Mean:", np.mean(a))
print("Maximum:", np.max(a))
print("Minimum:", np.min(a))
print("Standard Deviation:", np.std(a))
print("indexing:", a[4,2,5])
print("sclicing:", a[4:6,2:5,5:7])

