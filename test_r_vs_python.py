import numpy as np
from hapc.cv import pcghal_cv

np.random.seed(42)
n = 100
p = 1
X = np.random.uniform(-1, 1, (n, p))
Y = 2 * np.sin(8 * np.pi * (X[:, 0]**2)) / X[:, 0] + 10 + np.random.normal(0, 2, n)

Xnew = np.linspace(-1, 1, 10).reshape(-1, 1)

lambdas = np.exp(np.linspace(-3, 0, 10))
rescv = pcghal_cv(
    X, Y,
    maxdeg=1,
    npc=n,
    lambdas=lambdas,
    nfolds=3,
    max_iter=10,
    verbose=False,
    predict=Xnew,
    center=True
)

print("Python Predictions:")
print(rescv.predictions)
print("Python Best Lambda:")
print(rescv.best_lambda)
print("Python Alpha shape:", rescv.best_model.alpha.shape)
print("Python Alpha norm:", np.linalg.norm(rescv.best_model.alpha))
