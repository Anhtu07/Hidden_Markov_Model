import numpy as np
import gmmhmm

obs_1 = np.array([[-1, 2, 1], [1, 2, 1]])
obs_2 = np.array([[-1, 1], [1, 2]])

obs = np.array([obs_1])

mean_state_1 = np.array([[-2, 0], [0, 0]])
mean_state_2 = np.array([[0, 2], [2,2]])

mean = np.array([mean_state_1, mean_state_2])

cov = np.zeros((2, 2, 2, 2))

for i in range(2):
    for j in range(2):
        cov[i, j, :, :] = np.array([[1, 0], [0, 1]])

pi = np.array([0.5, 0.5])
A = np.array([[0.5, 0.5], [0.5, 0.5]])
c = np.array([[0.5, 0.5], [0.5, 0.5]])

m = gmmhmm.GMMHMM(2, 2)
m.mu = mean
m.covs = cov
m.c = c
m.fit(obs)