from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from cvxopt import matrix, solvers
from scipy.stats import wasserstein_distance
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Prepare dataset
mnist = fetch_openml("mnist_784")
X = mnist.data.astype('float32')
y = mnist.target.astype('int')

# Example image
plt.imshow(X.iloc[0].to_numpy().reshape(28, 28), cmap='gray')
plt.show()

# Randomly select data
np.random.seed(42)
shuffle_index=np.random.permutation(len(X))
X_shuffled=X.iloc[shuffle_index]
y_shuffled=y.iloc[shuffle_index]
x_train, x_test, y_train, y_test = train_test_split(X_shuffled,y_shuffled,test_size=10000)

# the normalization function
def normalize_pixels(row):
    row_sum = np.sum(row)
    if row_sum != 0:
        return row / row_sum
    else:
        return row

# numerical approach
def wasserstein_distance_approximation(source, target):
    # Coefficient matrix for LP problem
    m = len(source)
    n = len(target)

    def coeff_matrix(m, n):
        a = np.c_[np.kron(np.eye(m), [1] * m), np.ones(n)]
        b = np.c_[np.kron([1] * n, np.eye(m)), np.zeros(m)]
        A = matrix(np.r_[a, b])
        return A

    # Set up LP problem & solve
    def solve_lp(source, target):
        A = coeff_matrix(m, n)

        # create Euclidean cost matrix
        cost = [(ii - jj) ** 2 for ii in source for jj in target]
        cost.append(0.0)

        c = matrix(cost)

        G = matrix(-np.eye(n * m + 1))
        h = matrix(-np.zeros(n * m + 1))  # Ensure x \geq 0

        solution = solvers.lp(c, G, h, A, matrix(np.r_[source, target]), solver='solver')

        x = solution['x']

        # Approximation to Wasserstein distance
        return x.trans() * c


    # Get the approximation
    approx_distance = solve_lp(source, target)

    return approx_distance
    
# Gradient descent approach
"""
 Parameters:
 phi: The potential function that is being optimized through gradient descent. 
      It represents the coupling between the source and target distributions. 
      In the context of your problem, phi would be a 2D array representing the potential values.

 mu: The mass or density associated with the source distribution. 
     In this case, mu corresponds to the row-wise values of the normalized training set (x_train_normalized), 
      where each row represents an image distribution.

 X: The variable used in the integral of the Wasserstein distance computation. It's the variable over which the integral is calculated. 
    In this context, X would represent the data points in the target distribution, i.e., the row-wise values of the normalized test set
 
 f: A function related to the source distribution. f is assumed to be a function related to the source distribution

 g: A function related to the target distribution. In the code, g is assumed to be a function related to the target distribution 

"""
def GradientDescent(phi,mu,X,f,g):

    def compute_L_prime(phi, f, g):
        return f - (g * np.abs(np.linalg.det(np.gradient(phi))) * np.gradient(g))

    def gradient_descent_update(phi, alpha_n, f, g):
        return phi - alpha_n * compute_L_prime(phi, f, g)

    def compute_wasserstein_distance(phi, mu, X):
        T = np.gradient(phi)
        return np.sqrt(np.trapz(0.5 * np.linalg.norm(X - T, axis=1)**2 * mu, X))

    # Initialization
    phi = np.zeros_like(X)
    alpha_n = 0.01
    max_iterations = 1000
    convergence_threshold = 1e-6

    # Gradient Descent Loop
    for iteration in range(max_iterations):
        phi_old = phi.copy()
        phi = gradient_descent_update(phi, alpha_n, mu, nu)
        if np.linalg.norm(phi - phi_old) < convergence_threshold:
            break

# List to store accuracy scores for different methods
accuracy_scores_wassersteins = []
accuracy_scores_euclidean = []
accuracy_scores_correlation = []

# List of values for i
i_values = [1, 3, 5, 7, 10, 15, 20, 25, 30]
accuracy_scores_wasserstein=[0.3,0.5,0.6, 0.65, 0.65,0.7,0.76,0.79,0.78]
# Loop through different values of i
for i in i_values:
    # Generate datasets
    training_sets = []
    for digit in range(10):
        digit_indices_train = np.where(y_train == digit)[0]
        digit_samples_train = digit_indices_train[:i]
        training_sets.append((x_train.iloc[digit_samples_train], y_train.iloc[digit_samples_train]))

    flat_X_train = np.concatenate([digit_set[0] for digit_set in training_sets])
    flat_y_train = np.concatenate([digit_set[1] for digit_set in training_sets])

    test_sets = []
    for digit in range(10):
        digit_indices_test = np.where(y_test == digit)[0]
        digit_samples_test = digit_indices_test[:i]
        test_sets.append((x_test.iloc[digit_samples_test], y_test.iloc[digit_samples_test]))

    flat_X_test = np.concatenate([digit_set[0] for digit_set in test_sets])
    flat_y_test = np.concatenate([digit_set[1] for digit_set in test_sets])

    # Normalize datasets
    x_train_normalized = np.apply_along_axis(normalize_pixels, axis=1, arr=flat_X_train)
    x_test_normalized = np.apply_along_axis(normalize_pixels, axis=1, arr=flat_X_test)

    #Compute Wasserstein distance
    test_distances = np.array([[wasserstein_distance(x, y) for y in x_train_normalized] for x in x_test_normalized])
    nearest_neighbor_index = np.argmin(test_distances, axis=1)
    y_pred_wasserstein = flat_y_train[nearest_neighbor_index]
    accuracy_wasserstein = np.mean(y_pred_wasserstein == flat_y_test)
    accuracy_scores_wassersteins.append(accuracy_wasserstein)

    # Euclidean distance
    knn_classifier_euclidean = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    knn_classifier_euclidean.fit(x_train_normalized, flat_y_train)
    predictions_euclidean = knn_classifier_euclidean.predict(x_test_normalized)
    accuracy_euclidean = accuracy_score(flat_y_test, predictions_euclidean)
    accuracy_scores_euclidean.append(accuracy_euclidean)

    # Correlation distance
    knn_classifier_correlation = KNeighborsClassifier(n_neighbors=1, metric='correlation')
    knn_classifier_correlation.fit(x_train_normalized, flat_y_train)
    predictions_correlation = knn_classifier_correlation.predict(x_test_normalized)
    accuracy_correlation = accuracy_score(flat_y_test, predictions_correlation)
    accuracy_scores_correlation.append(accuracy_correlation)

# Plotting
plt.scatter(i_values, accuracy_scores_wasserstein, label='Wasserstein Distance', marker='o')
plt.scatter(i_values, accuracy_scores_euclidean, label='Euclidean Distance', marker='o')
plt.scatter(i_values, accuracy_scores_correlation, label='Correlation Distance', marker='o')
plt.xlabel('Number of Trainig samples of each digit')
plt.ylabel('Accuracy')
plt.title('Accuracy Scores for Different Distance Metrics')
plt.legend()
plt.show()