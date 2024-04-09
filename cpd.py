# Import necessary libraries
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import multivariate_normal

# Generate synthetic data
np.random.seed(0)
N = 100  # Number of points in X
M = 120  # Number of points in Y
d = 2  # Dimensionality of points
noise_sigma = 0.05  # Noise level

# Generate point clouds
X = np.random.rand(N, d)
Y = np.random.rand(M, d)

# Apply random transformation to Y
theta = np.pi / 4  # Rotation angle
t = np.array([0.5, 0.5])  # Translation vector
R = np.array([[np.cos(theta), -np.sin(theta)],
             [np.sin(theta), np.cos(theta)]])  # Rotation matrix
Y = np.dot(Y, R.T) + t + np.random.randn(M, d) * noise_sigma

# Initialize CPD parameters
# You may need to tune these parameters for your specific application
num_iter = 50  # Number of iterations
lambda_ = 2  # Regularization parameter
beta = 1  # Scaling parameter for Gaussian kernel
tol = 1e-3  # Tolerance for convergence

# Initialize transformation
T = np.eye(d + 1)

# Initialize GMM parameters
pi = np.ones(M) / M  # Mixing coefficients
mu = np.copy(Y)  # Means
Sigma = np.tile(np.eye(d), (M, 1, 1))  # Covariance matrices

# Main loop
for iter in range(num_iter):
    # Expectation step
    # Compute soft correspondences
    P = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            P[i, j] = pi[j] * multivariate_normal.pdf(X[i], mean=mu[j], cov=Sigma[j])
    P /= np.sum(P, axis=1, keepdims=True)

    # Maximization step
    # Update transformation
    ones_X = np.ones((N, 1))  # Column vector with N rows
    ones_Y = np.ones((M, 1))  # Column vector with M rows
    X_h = np.concatenate((X, ones_X), axis=1)
    Y_h = np.concatenate((Y, ones_Y), axis=1)

    # Compute intermediate matrices
    A = np.dot(X_h.T, np.diag(np.sum(P, axis=1)))
    B = np.dot(X_h.T, P)
    C = np.dot(B, Y_h)

    # Solve linear system using SVD
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    R = np.dot(Vt.T, U.T)
    T = np.zeros((d + 1, d + 1))
    T[:d, :d] = R
    T[:d, -1] = np.mean(Y, axis=0) - np.dot(R, np.mean(X, axis=0))

    # Update GMM parameters
    pi = np.mean(P, axis=0)
    mu = np.dot(P.T, X) / np.sum(P, axis=0, keepdims=True).T
    for j in range(M):
        diff = X - mu[j]
        Sigma[j] = np.dot(diff.T, np.dot(np.diag(P[:, j]), diff)) / np.sum(P[:, j])

    # Check for convergence
    if iter > 0 and np.abs(np.log(np.sum(P_old)) - np.log(np.sum(P))) < tol:
        break

    # Update P_old for next iteration
    P_old = np.copy(P)

# Apply transformation to X
X_transformed = np.dot(X_h, T.T)[:, :2]

# Plot results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c='b', label='Original X')
plt.scatter(Y[:, 0], Y[:, 1], c='r', label='Original Y')
plt.title('Original Point Clouds')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c='b', label='Transformed X')
plt.scatter(Y[:, 0], Y[:, 1], c='r', label='Original Y')
plt.title('Aligned Point Clouds')
plt.legend()

plt.show()
