import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create dataset: damaged = 1, not damaged = 0
def create_data(n=1000):
    damaged = np.random.randn(n // 2, 2) + 7  # Damaged samples
    normal = np.random.randn(n // 2, 2) + 2   # Normal samples
    X = np.vstack((damaged, normal))
    y = np.array([1] * (n // 2) + [0] * (n // 2))
    return X, y

# Simple AIS Classifier
class SimpleAIS:
    def __init__(self, detectors=100, threshold=1.5):
        self.detectors = []
        self.n = detectors
        self.threshold = threshold

    def train(self, X, y):
        normal = X[y == 0]
        while len(self.detectors) < self.n:
            d = np.random.uniform(0, 10, 2)
            if all(np.linalg.norm(d - n) > self.threshold for n in normal):
                self.detectors.append(d)

    def predict(self, X):
        return [1 if any(np.linalg.norm(x - d) < self.threshold for d in self.detectors) else 0 for x in X]

# Run the AIS
X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = SimpleAIS(detectors=100, threshold=1.8)
model.train(X_train, y_train)
y_pred = model.predict(X_test)

# Results
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Number of detectors generated: {len(model.detectors)}")
