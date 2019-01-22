import numpy as np
from scipy.stats import multivariate_normal
def mixture_pdf(X, c, means, covs, component=None):
    '''
    # Return the pdf of GMM 
    # @params:
    # X: array-like
    # c: array-like
    # means: matrox with shape (shape(c), shape(X))
    # cov: tensor with shape (shape(c), shape(X), shape(X))
    # component: to calculate the responsibility wrt the gmm
    '''
    density = 0
    for i in range(len(c)):
        density += c[i]*multivariate_normal.pdf(X, means[i, :].T, covs[i, :, :].T)
    if component is None:
        return density
    else:
        return c[component]*multivariate_normal.pdf(X, means[component, :].T, covs[component, :, :].T)/density

class GMMHMM:
    def __init__(self, n_states, n_mix):
        self.n_states = n_states
        self.n_mix = n_mix
        # Init
        #self.transmat_ = np.ones((n_states, n_states))/n_states
        self.startprob_ = np.ones(n_states)/n_states
        self.transmat_ = np.ones((n_states, n_states))/n_states
        
        # self.mu: mean tensor, shape(n_states, n_mix, n_features)
        self.mu = None
        # self.covs: cov tensor, shape(n_states, n_mix, n_features, n_features)
        self.covs = None
        # self.c: mixture array, shape(n_states, n_mix)
        self.c = None
        self.n_features = None
        self.n_iter = 0
        
    def _forward(self, B):
        '''
        # Do forward step
        # @params
        # B: the density matrix given the observation, shape(n_states, time)
        '''
        log_likelihood = 0
        T = B.shape[1]
        # alpha: shape(n_states, obs_length)
        alpha = np.zeros((self.n_states, T))
        for t in range(T):
            if t == 0:
                alpha[:, t] = self.startprob_*B[:, t]
            else:
                alpha[:, t] = np.dot(self.transmat_.T, alpha[:, t-1])*B[:, t]
            
            # Normalize 
            alpha_sum = np.sum(alpha[:, t])
            alpha[:, t] /= alpha_sum
            log_likelihood = log_likelihood + np.log(alpha_sum)
        return log_likelihood, alpha
    
    def _backward(self, B):
        '''
        # Do backward step
        # @params:
        # B: the density matrix given the observation, shape(n_states, time)
        '''
        T = B.shape[1]
        # beta: shape(n_states, obs_length)
        beta = np.zeros((self.n_states, T))
        for t in range(T-1, -1, -1):
            if t == T-1:
                beta[:, t] = np.ones(B.shape[0])
            else:
                beta[:, t] = np.dot(self.transmat_, (B[:, t+1]*beta[:, t+1]))
                # Normalize
                beta[:, t] /= np.sum(beta[:, t])
        return beta
    
    def _state_likelihood(self, obs):
        '''
        # Calculate the emission probability
        # @params:
        # obs: observation in 2d (n_features, time)
        # @return:
        # B: B matrix shape (n_states, time)
        '''
        obs = np.atleast_2d(obs)
        B = np.zeros((self.n_states, obs.shape[-1]))
        for s in range(self.n_states):
            B[s, :] = mixture_pdf(obs.T, self.c[s, :], self.mu[s, :, :], self.covs[s, :, :, :])
        return B
        
    def _em_init(self, obs):
        '''
        # Init n_features, mu, cov, c with respect to data
        # @params:
        # obs: observation in 3d (n_observations, n_features, time)
        '''
        T = obs.shape[-1]
        if self.n_features is None:
            self.n_features = obs.shape[1]
        if self.mu is None:
            self.mu = np.zeros((self.n_states, self.n_mix, self.n_features))
            for state in self.n_states:
                rand_idxs = np.random.choice(T, size = self.n_mix, replace = False)
                self.mu[state, :, :] = obs[0, :, rand_idxs]
        if self.covs is None:
            self.covs = np.zeros((self.n_states, self.n_mix, self.n_features, self.n_features))
            for state in self.n_states:
                for mix in self.n_mix:
                    self.covs[state, mix, :, :] = np.diag(np.ones(self.n_features))
        if self.c is None:
            self.c = np.ones((self.n_states, self.n_mix))/self.n_mix
    
    def fit(self, obs):
        '''
        # fit 
        # @params:
        # obs: observation in 3d (n_observations, n_features, time)
        '''
        self._em_init(obs)
        prev_score = None
        curr_score = None
        threshold = 10
        while( prev_score is None or curr_score is None or np.abs(prev_score-curr_score)>threshold):
            self._em(obs)
            prev_score = curr_score
            curr_score = 0
            self.n_iter += 1
            for o in obs:
                curr_score += self.score(o)
            curr_score = curr_score/obs.shape[0]
            print(curr_score)
                
    def _em(self, obs):
        '''
        # Calculate the xi and gamma
        # @params:
        # obs: observation in 3d (n_observations, n_features, time)
        '''
        T = obs.shape[-1]
        n_features = obs.shape[1]
        n_observations = obs.shape[0]
        # Calculate B for the observation data
        gamma = np.zeros((n_observations, self.n_states, self.n_mix, T))
        # Xi is calculated as a sum of all times and all observation
        xi = np.zeros((n_observations, self.n_states, self.n_states, T))
        xi_sum = np.zeros((self.n_states, self.n_states))
        for observation_idx in range(n_observations):
            current_observation = obs[observation_idx, :, :]
            B = self._state_likelihood(current_observation)
            log_likelihood, alpha = self._forward(B)
            beta = self._backward(B)
            # care, myboi
            gamma_1 = alpha*beta/np.diag(np.dot(alpha.T,beta))
            for k in range(self.n_mix):
                gamma[observation_idx, :, k, :] = gamma_1
            for j in range(self.n_states):
                for k in range(self.n_mix):
                    for t in range(T):
                        # print(mixture_pdf(current_observation[:, t], self.c[j, :], self.mu[j, :], self.covs[j, :, :, :], component=k))
                        gamma[observation_idx, j, k, t] = alpha[j, t]*beta[j, t]/np.diag(np.dot(alpha.T,beta))[t]*mixture_pdf(current_observation[:, t], self.c[j, :], self.mu[j, :], self.covs[j, :, :, :], component=k)
        
        
            for t in range(T-1):
                '''
                partial_sum = self.transmat_ * np.dot(alpha[:, t], (beta[:, t] * B[:, t + 1]).T)
                # How about p_s = 0??
                xi_sum += partial_sum/np.sum(partial_sum)
                '''
                sum_i_j = 0
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[observation_idx, i, j, t] = alpha[i, j]*self.transmat_[i, j]*beta[i,j]*B[j, t+1]
                        sum_i_j += xi[observation_idx, i, j, t]
                        
                xi[:, :, :, t] /= sum_i_j
        
        
        xi_sum_over_obs = np.sum(xi, axis=0)
        xi_sum_over_obs_t = np.sum(xi_sum_over_obs,axis=2)
        xi_sum_over_obs_t_j = np.sum(xi_sum_over_obs_t, axis=1)
        #xi_sum_over_j = np.sum(xi_sum, axis = 1)
        expected_transmat = xi_sum_over_obs_t/xi_sum_over_obs_t_j[None, :].T
        
        
        expected_mu = np.zeros((self.n_states, self.n_mix, self.n_features))
        expected_covs = np.zeros((self.n_states, self.n_mix, self.n_features, self.n_features))
        expected_c = np.zeros((self.n_states, self.n_mix))
        
        gamma_sum_over_t = np.sum(gamma, axis=3)
        gamma_sum_over_t_obs = np.sum(gamma_sum_over_t, axis = 0)
        gamma_sum_over_t_obs_mix = np.sum(gamma_sum_over_t_obs, axis=1)
        
        expected_c = gamma_sum_over_t_obs/gamma_sum_over_t_obs_mix[None, :].T
        gamma_sum_over_obs_mix = np.sum(np.sum(gamma[:, :, :, 0], axis=2), axis=0)
        expected_pi = gamma_sum_over_obs_mix/n_observations
        
        for obs_idx in range(n_observations):
            for t in range(T):
                p = np.zeros((self.n_states, self.n_mix, self.n_features))
                
                for s in range(self.n_states):
                    for m in range(self.n_mix):
                        expected_mu[s, m, :] += gamma[obs_idx, s, m, t]*obs[obs_idx, :, t]  
                        
                        diff = obs[obs_idx][:, t] - self.mu[s, m, :]
                        expected_covs[s, m, :, :] += gamma[obs_idx, s, m, t]*np.dot(diff[None, :].T, diff[None, :])
                
        for f_idx in range(self.n_features):
            expected_mu[:, :, f_idx] /= gamma_sum_over_t_obs
        
        for f_idx in range(self.n_features):
            for f_idx2 in range(self.n_features):
                expected_covs[:, :, f_idx, f_idx2] /= gamma_sum_over_t_obs
                
        self.c = expected_c
        self.covs = expected_covs
        self.mu = expected_mu
        self.startprob_ = expected_pi
        self.transmat_ = expected_transmat
        
    def score(self, obs):
        '''
        # @
        '''
        B = self._state_likelihood(obs)
        return self._forward(B)[0]
            
        