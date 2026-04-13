import numpy as np

def normalize_priors(priors):
    if np.sum(priors) == 0:
        return np.zeros_like(priors)
    return priors / np.sum(priors)

def best_response(x, thresholds, priors, c):
    posteriors = normalize_priors(priors)
    search_space = [x] + thresholds[thresholds > x].tolist()
    utilities = []
    for i, x_p in enumerate(search_space):
        utility = np.dot(posteriors, x_p >= thresholds)
        cost = c * abs(x-x_p)
        utilities.append(utility - cost + (i*1e-6))
    return search_space[np.argmax(utilities)]

def best_response_vectorized(X, thresholds, priors, c):
    posteriors = normalize_priors(priors)
    utilities_expected = np.array([
        np.dot(posteriors, thresholds[j] >= thresholds)
        for j in range(len(thresholds))
    ])

    pass_matrix = X[:, None] >= thresholds[None, :]
    utility_stay = pass_matrix @ posteriors

    X_col = X[:, None]
    thresholds_row = thresholds[None, :]

    feasible = thresholds_row > X_col
    utility_jump = utilities_expected[None, :] - c * np.abs(thresholds_row - X_col)
    utility_jump[~feasible] = -np.inf

    utilities = np.concatenate([utility_stay[:, None], utility_jump],axis=1)

    best_idx = np.argmax(utilities + np.array([i*1e-6 for i in range(utilities.shape[1])]), axis=1)
    X_p = np.where(best_idx == 0, X, thresholds[best_idx - 1])

    return X_p

def accuracy_loss(X, X_p, thresholds, priors, threshold_true):
    Y_true = (X >= threshold_true).astype(float)
    losses = []
    for threshold in thresholds:
        Y_p = (X_p >= threshold).astype(float)
        acc_loss = np.abs(Y_true - Y_p).mean()
        losses.append(acc_loss)
    losses = np.array(losses)
    posteriors = normalize_priors(priors)
    return np.dot(losses, posteriors)

def accuracy_loss_vectorized(X, X_p, thresholds, priors, threshold_true):
    Y_true = (X >= threshold_true).astype(float)
    Y_p = (X_p[:, None] >= thresholds[None, :]).astype(float)
    losses = np.abs(Y_true[:, None] - Y_p).mean(axis=0)
    posteriors = normalize_priors(priors)
    return np.dot(losses, posteriors)

def evaluate_partition(X, partition, thresholds, priors, threshold_true, c, return_all=False):
    thresholds_p = thresholds[partition]
    priors_p = priors[partition]
    X_p = best_response_vectorized(X, thresholds_p, priors_p, c)
    acc_loss_p = accuracy_loss_vectorized(X, X_p, thresholds_p, priors_p, threshold_true)
    if return_all:
        return acc_loss_p, thresholds_p, priors_p
    return acc_loss_p

def evaluate_system(X, partitions, thresholds, priors, threshold_true, c):
    acc_loss = 0.
    for partition in partitions:
        acc_loss_p = evaluate_partition(X, partition, thresholds, priors, threshold_true, c)
        acc_loss += acc_loss_p * np.sum(priors[partition])
    return acc_loss