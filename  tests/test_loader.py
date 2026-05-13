import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
X, y = load_iris(return_X_y=True)

# Correct split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Validation checks
assert len(X_train) > 0
assert len(X_test) > 0
assert len(np.unique(y_train)) == len(np.unique(y))
assert len(np.unique(y_test)) == len(np.unique(y))

# Print dataset sizes
print("Train size:", len(X_train))
print("Test size:", len(X_test))

# Visualization
classes = np.unique(y)
train_counts = [np.sum(y_train == c) for c in classes]
test_counts = [np.sum(y_test == c) for c in classes]

x = np.arange(len(classes))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, train_counts, width, label='Train')
plt.bar(x + width/2, test_counts, width, label='Test')

plt.xticks(x, ['Setosa', 'Versicolor', 'Virginica'])
plt.ylabel('Count')
plt.title('Train/Test Distribution')
plt.legend()

plt.show()
