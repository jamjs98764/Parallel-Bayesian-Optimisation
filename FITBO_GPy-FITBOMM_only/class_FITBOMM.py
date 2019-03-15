#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:45:16 2017

@author: robin
"""
import os
import pickle
import numpy as np
import scipy as sp
from scipy import stats
import copy 
from scipy.optimize import minimize,fmin_l_bfgs_b
import GPy
from MCMC_Sampler import elliptical_slice
from Test_Funcs import branin,func1D
from utilities import Gaussianprior,GMMpdf,computeEntropyGMM1Dquad
from GPy.util.linalg import pdinv, dpotrs, tdot
import batch_proposals
import GP_models
from scipy.stats import norm

class Bayes_opt():
    def __init__(self, func, lb, ub, var_noise, input_type = [0]):
        self.func = func
        self.lb = lb
        self.ub = ub
        self.var_noise = var_noise
        self.ntest = 2000
        self.input_type = input_type

    def initialise(self, X_init=None, Y_init=None, kernel=None):
        assert X_init.ndim == 2, "X_init has to be 2D array"
        assert Y_init.ndim == 2, "Y_init has to be 2D array"
        self.X_init = X_init
        self.Y_init = Y_init
        self.X = X_init
        self.Y = Y_init

        # Input dimension
        self.X_dim = self.X.shape[1]

        # Find min observed functional value and its location
        self.arg_opt = np.atleast_2d(self.X[np.argmin(self.Y)])
        if kernel == None:
            self.kernel = GPy.kern.RBF(input_dim=self.X_dim, ARD=True)
        else:
            self.kernel = kernel

    def _log_likelihood(self, log_params):
        # Returns log likelihood, p(D|hyperparams)
        params = np.exp(log_params)
        l_scales = params[0:self.X_dim]
        output_var = params[self.X_dim] # Vertical length scale
        noise_var = params[self.X_dim + 1] 
        # compute eta
        eta = np.min(self.Y) - params[self.X_dim + 2] # QUESTION: what is this?
        # compute the observed value for g instead of y
        g_ob = np.sqrt(2.0 * (self.Y - eta))

        kernel = GPy.kern.RBF(input_dim=self.X_dim, ARD=True, variance=output_var, lengthscale=l_scales)
        Kng = kernel.K(self.X)
        # QUESTION: does not seem to follow conditional variance form in eqn 6
        
        # compute posterior mean distribution for g TODO update this
        # GPg = GPy.models.GPRegression(self.X, g_ob, kernel, noise_var=1e-8)
        # mg,_ = GPg.predict(self.X)
        mg = g_ob

        # approximate covariance matrix of y using linearisation technique
        Kny = mg * Kng * mg.T + (noise_var+1e-8) * np.eye(Kng.shape[0])

        # compute likelihood terms
        Wi, LW, LWi, W_logdet = pdinv(Kny) # from GPy module
        # Wi = inverse of Kny (ndarray)
        # LW = Cholesky decomposition of Kny (ndarray)
        # LWi = Cholesky decomposition of inverse of Kny (ndarray)
        # W_logdet = log determinant of Kny (float)
        
        alpha, _ = dpotrs(LW, self.Y, lower=1)
        loglikelihood = 0.5 * (-self.Y.size * np.log(2 * np.pi) - self.Y.shape[1] * W_logdet - np.sum(alpha * self.Y))
        # Log marginal likelihood for GP, based on Rasmussen eqn 2.30
        
        return loglikelihood

    def _log_posterior(self, log_params, mean_ln_yminob_minus_eta, var_ln_yminob_minus_eta):
        # Returns posterior, p(y|D, hyperparams, eta)
        # QUESTION: is this true? What is the formula?
        
        params = np.exp(log_params)
        # compute log likelihood
        log_likelihood = self._log_likelihood(log_params)

        # log prior of hyperparameters
        det_l_scales = np.product(params[0:self.X_dim])
        logl_priormu = np.log(0.3 * np.ones(self.X_dim))
        logl_priorvar = np.diag(3.0 * np.ones(self.X_dim))
        prior_l_scales = Gaussianprior(log_params[0: self.X_dim], logl_priormu, logl_priorvar) / det_l_scales
        # Gaussianprior(X, mean, variance)
        # Returns pdf of X based on multivariate Gaussian with given mean and variance
        prior_output_var = Gaussianprior(log_params[self.X_dim], np.log(1.0), 3.0) / params[self.X_dim]
        prior_noise_var = Gaussianprior(log_params[self.X_dim + 1], np.log(0.01), 1.0) / params[self.X_dim + 1]
        prior_ymin_eta = Gaussianprior(log_params[self.X_dim + 2], mean_ln_yminob_minus_eta, var_ln_yminob_minus_eta )/\
                         params[self.X_dim + 2]
        log_posterior = log_likelihood + np.log(prior_l_scales * prior_output_var * prior_noise_var * prior_ymin_eta)
        return log_posterior  # returns a scalar value

    def _samplehyper(self, mean_ln_yminob_minus_eta, var_ln_yminob_minus_eta):
        
        Nsamples = self.burnin + self.mc_samples
        L_d = np.ones(self.X_dim)
        # define initial guess for hyperparameters
        init = np.hstack((np.log(0.3*L_d), np.log(10.0), np.log(1e-2), mean_ln_yminob_minus_eta))

        prior_mu = np.zeros(self.X_dim+3)
        prior_cov = np.diag(1.0 * np.ones(self.X_dim+3))
        prior_cov_chol = sp.linalg.cholesky(prior_cov, lower=True)

        params_array = init
        log_params = np.zeros((Nsamples, len(params_array)))
        sampler_options = {"cur_log_like": None, "angle_range": 0}
        extra_para = (mean_ln_yminob_minus_eta, var_ln_yminob_minus_eta)
        for i in range(Nsamples):
            params_array, current_ll = elliptical_slice(
                params_array,
                self._log_posterior,
                prior_cov_chol,
                prior_mu,
                * extra_para,
                **sampler_options)
            log_params[i,:] = params_array.ravel()
            current_ll = current_ll  # for diagnostics QUESTION: what is this for?
        self.params = np.exp(log_params[self.burnin:, :])

    def _FITBOMM(self,x):
        '''FITBO-Moment Matching acquisition function'''
        x = np.atleast_2d(x)
        noise_param = self.params[:, self.X_dim + 1]
        eta = np.min(self.Y)- self.params[:, self.X_dim + 2]
        N_hypsamples = len(self.params)

        n = x.shape[0]  # number of test point
        Mean_y = np.zeros([N_hypsamples, n])
        Var_y = np.zeros([N_hypsamples, n])
        Entropy_T2 = np.zeros([N_hypsamples, n])
        weights = 1.0 / N_hypsamples

        for i in range(N_hypsamples):

            mg, varg = self.GP[i].predict(x)
            my = 0.5 * (mg ** 2.0) + eta[i]
            vary = (mg ** 2) * varg + noise_param[i]
            Mean_y[i, :] = my[:, 0]
            Var_y[i, :] = vary[:, 0]
            # Compute the 2nd entropy term ( entropy of a gaussian )
            Entropy_T2[i, :] = 0.5 * np.log(2.0 * np.pi * self.e * vary)[:, 0]

        # Compute the 1st entropy term in FITBOMM ( entropy of a gaussian mixture )
        mean_GMM = weights * np.sum(Mean_y, axis=0)
        cov_GMM = weights * np.sum(Var_y + Mean_y ** 2, axis=0) - mean_GMM ** 2
        Entropy_1 = 0.5 * np.log(2.0 * np.pi * self.e * cov_GMM)
        # Compute the acquisition function

        Mutual_info = Entropy_1 - np.mean(Entropy_T2, axis=0)
        # return - Mutual_info for minimiser
        return - Mutual_info

    def _FITBO(self,x):
        '''FITBO-Numerical Quadrature acquisition function'''
        x = np.atleast_2d(x)
        noise_param = self.params[:, self.X_dim + 1]
        eta = np.min(self.Y) - self.params[:, self.X_dim + 2]
        N_hypsamples = len(self.params)

        n = x.shape[0]  # number of test point
        Mean_y = np.zeros([N_hypsamples, n])
        Var_y = np.zeros([N_hypsamples, n])
        Entropy_T2 = np.zeros([N_hypsamples, n])
        weights = 1.0 / N_hypsamples

        for i in range(N_hypsamples):

            mg, varg = self.GP[i].predict(x)
            my = 0.5 * (mg ** 2.0) + eta[i]
            vary = (mg ** 2) * varg + noise_param[i]
            Mean_y[i, :] = my[:, 0]
            Var_y[i, :] = vary[:, 0]
            # Compute the 2nd entropy term ( entropy of a gaussian )
            Entropy_T2[i, :] = 0.5 * np.log(2.0 * np.pi * self.e * vary)[:, 0]

        # Compute the 1st entropy term in FITBOMM ( entropy of a gaussian mixture )
        weights = 1 / N_hypsamples
        Entropy_1 = np.zeros(n)

        for j2 in range(n):
            Entropy_1[j2] = computeEntropyGMM1Dquad(Mean_y[:, j2], Var_y[:, j2], weights, 3)

        # Compute the acquisition function
        Mutual_info = Entropy_1 - np.mean(Entropy_T2, axis=0)
        # return - Mutual_info for minimiser
        return - Mutual_info
    
    def _fit_GP(self):
        '''collect GPs defined using observed g values and all hyperparameter samples'''
        self.GP = [] # Contains a collection of GPs based on hyperparameter sampling
        self.kernel = [] # Contains a collection of kernels based on hyperparameter sampling
        lscale_param = self.params[:, 0:self.X_dim]
        var_param = self.params[:, self.X_dim]
        noise_param = self.params[:, self.X_dim + 1]
        eta = np.min(self.Y) - self.params[:, self.X_dim + 2] # There are 

        for i in range(len(self.params)): # iterate across each MC sample of hyperparaters (Nsamples = 50)
            diff = self.Y - eta[i]
            diff.clip(min=0) # Change negative values to zero
            self.G = np.sqrt(2.0 * diff)
            self.kernel.append(GPy.kern.RBF(input_dim=self.X_dim, ARD=True, variance=var_param[i], lengthscale=lscale_param[i, :]))
            self.GP.append(GPy.models.GPRegression(self.X, self.G, self.kernel[i], noise_var=(noise_param[i])))


    def _fit_GP_normal(self):
        '''collect GPs defined using observed y values and all hyperparameter samples'''
        self.GP_normal = []
        noise_param = self.params[:, self.X_dim + 1]
        for i in range(len(self.params)):
            self.GP_normal.append(GPy.models.GPRegression(self.X, self.Y, self.kernel[i], noise_var=noise_param[i]))


    def _marginalised_posterior_mean(self, x):
        '''Marginalize GP-normal over all hyperparam sample'''
        x = np.atleast_2d(x) # Wrap array x into 2D-array
        n = x.shape[0]
        Meanf = np.zeros([len(self.params), n])
        # dMeanf = np.zeros([len(self.params), n])
        for i in range(len(self.params)):
            my, vary = self.GP_normal[i].predict(x) # GPy.models.predict returns (mean, variance) of prediction
            # dmy, dvary = self.GP_normal[i].predictve_gradients(x)

            Meanf[i, :] = my[:, 0]
            # dMeanf[i, :] = dmy[:, 0]
        pos_mean = np.mean(Meanf, axis=0)
        # dpos_mean = np.mean(dMeanf, axis=0)
        return pos_mean
    
    def _marginalised_posterior_var(self, x):
        '''Marginalize GP-normal over all hyperparam sample'''
        x = np.atleast_2d(x) # Wrap array x into 2D-array
        n = x.shape[0]
        Mean_var = np.zeros([len(self.params), n])
        for i in range(len(self.params)):
            my, vary = self.GP_normal[i].predict(x) # GPy.models.predict returns (mean, variance) of prediction
            Mean_var[i, :] = vary[:, 0]
        pos_var = np.mean(Mean_var, axis=0)
        return pos_var
    
    def _store_full_posterior_mean_var(self):
        ''' Saves the posterior mean and variance of GP model at each iteration'''
        Xgrid = np.random.uniform(0.0, 1.0, (self.ntest, self.X_dim))
        full_mean = []
        full_var = []
        
        for x in Xgrid:
            pos_mean = self._marginalised_posterior_mean(x)
            pos_var = self._marginalised_posterior_var(x)
            full_mean.append(pos_mean)
            full_var.append(pos_var)
            
        return full_mean, full_var       

    def _gloabl_minimser(self,func):
        ntest = 2000;
        Xgrid = np.random.uniform(0.0, 1.0, (ntest, self.X_dim))
        Xtest = np.vstack((Xgrid,self.X))
        Func_value_test = func(Xtest)
        idx_test = np.argmin(Func_value_test)
        X_start = Xtest[idx_test, :]
        # bnds = ((0.0, 1.0), (0.0, 1.0))
        bnds = tuple((li, ui) for li, ui in zip(self.lb, self.ub))

        res = minimize(func, X_start, method='L-BFGS-B', jac=False, bounds=bnds)
        x_opt = res.x[None, :]
        
        # Rounding
        for i in range(len(self.input_type)):
            if self.input_type[i] == True:
                x_opt[0][i] = np.rint(x_opt[0][i])
        
        return x_opt

    def iteration_step(self, iterations, mc_burn , mc_samples,bo_method, \
                       seed, resample_interval, dir_name = 'Exp_Data/'):
        
        # Array to store mean GP values at each iteration
        self.full_mean_record = np.zeros([iterations, self.ntest])
        self.full_var_record = np.zeros([iterations, self.ntest])
        self.full_PI_value = np.zeros([1, iterations])
        
        np.random.seed(seed)

        X_optimum = np.atleast_2d(self.arg_opt)
        Y_optimum = np.atleast_2d(np.min(self.Y))
        X_for_L2  = X_optimum
        Y_for_IR  = Y_optimum

        # sample hyperparameters
        self.mc_samples = mc_samples  # number of samples
        self.burnin = mc_burn         # number of burnin
        self.e = np.exp(1)
        # initial guess for log(y_min - eta_hyparameters)
        mean_log_ymin_minus_eta = np.log(1.0)
        var_log_ymin_minus_eta = 0.1
        self._samplehyper(mean_log_ymin_minus_eta, var_log_ymin_minus_eta)
        mean_log_ymin_minus_eta, var_log_ymin_minus_eta = sp.stats.norm.fit(np.log(self.params[:, self.X_dim + 2])) # Fit a normal distribution to sampled etas

        # fit GP models to hyperparameter samples
        self._fit_GP() 

        # Specify acquisition function
        if bo_method == 'FITBOMM':
            acqu_func = self._FITBOMM
        else:
            acqu_func = self._FITBO
            
        
        for k in range(iterations):

            # np.random.seed(seed*100)
            # optimise the acquisition function to get the next query point and evaluate at next query point
            current_y_best = self.Y.min()
            x_next = self._gloabl_minimser(acqu_func)
            max_acqu_value = - acqu_func(x_next)
            y_next = self.func(x_next) + np.random.normal(0, self.var_noise, len(x_next)) # Query x_next, but y = f(x) + noise

            # update the observation data
            self.X = np.vstack((self.X, x_next))
            self.Y = np.vstack((self.Y, y_next))

            # resample hyperparameters
            if k % resample_interval ==0:
                self._samplehyper(mean_log_ymin_minus_eta, var_log_ymin_minus_eta)
                mean_log_ymin_minus_eta, var_log_ymin_minus_eta = sp.stats.norm.fit(
                                                                np.log(self.params[:, self.X_dim + 2]))

            # update the GP models and the minimum observed value
            self._fit_GP()

            # optimise the marginalised posterior mean to get the prediction for the global optimum/optimiser
            self._fit_GP_normal()
            x_opt = self._gloabl_minimser(self._marginalised_posterior_mean)
            y_opt = self.func(x_opt)
            X_optimum = np.concatenate((X_optimum, np.atleast_2d(x_opt)))
            Y_optimum = np.concatenate((Y_optimum, np.atleast_2d(y_opt)))
            X_for_L2 = np.concatenate((X_for_L2, np.atleast_2d(X_optimum[np.argmin(Y_optimum),:])))
            Y_for_IR = np.concatenate((Y_for_IR, np.atleast_2d(min(Y_optimum))))

            print("bo:"+ bo_method + ",seed:{seed},itr:{iteration},x_next: {next_query_loc},y_next:{next_query_value}, acq value: {best_acquisition_value},"
                "x_opt:{x_opt_pred},y_opt:{y_opt_pred}"
                .format(seed = seed,
                        iteration=k,
                        next_query_loc=x_next,
                        next_query_value=y_next,
                        best_acquisition_value=max_acqu_value,
                        x_opt_pred=X_for_L2[-1,:], # QUESTION: why is this always the last value?
                        y_opt_pred=Y_for_IR[-1,:]
                        ))
            
            # Saving GP values for PI calculation
            x_next_mean = self._marginalised_posterior_mean(x_next)
            x_next_var = self._marginalised_posterior_var(x_next)
            PI_value = norm.cdf((current_y_best - (x_next_mean)) / np.sqrt(x_next_var))
            print("pi")
            print(PI_value)
            self.full_PI_value[:, k] = PI_value

            """
            full_mean, full_var = self._store_full_posterior_mean_var()
            self.full_mean_record[k,:] = full_mean
            self.full_var_record[k,:] = full_var
            print(self.full_mean_record)
            """
            
        # Just for saving
        new_dir = dir_name + str(seed) + '_seed/' 
        
        try:
            os.mkdir(new_dir)
        except FileExistsError:
            pass

        file_name = new_dir + 'sequential,intermediate_vars.pickle'
        
        pickle_dict = {
                "X": self.X, 
                "Y": self.Y, 
                "X_init": self.X_init,
                "Y_init": self.Y_init,
                "PI_values": self.full_PI_value
                }
        
        with open(file_name, 'wb') as f:
            pickle.dump(pickle_dict, f)          

        return X_for_L2, Y_for_IR


#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################

class Bayes_opt_batch():
    def __init__(self, func, lb, ub, var_noise, input_type = [0]):
        self.func = func
        self.lb = lb
        self.ub = ub
        self.var_noise = var_noise
        self.ntest = 2000
        self.input_type = input_type

    def initialise(self, X_init=None, Y_init=None, kernel=None):
        assert X_init.ndim == 2, "X_init has to be 2D array"
        assert Y_init.ndim == 2, "Y_init has to be 2D array"
        self.X_init = X_init
        self.Y_init = Y_init
        self.X = X_init
        self.Y = Y_init

        # Input dimension
        self.X_dim = self.X.shape[1]

        # Find min observed functional value and its location
        self.arg_opt = np.atleast_2d(self.X[np.argmin(self.Y)])
        if kernel == None:
            self.kernel = GPy.kern.RBF(input_dim=self.X_dim, ARD=True)
        else:
            self.kernel = kernel

    def _log_likelihood(self, log_params):
        # Returns log likelihood, p(D|hyperparams)
        params = np.exp(log_params)
        l_scales = params[0:self.X_dim]
        output_var = params[self.X_dim] # QUESTION: difference between output and noise variance
        noise_var = params[self.X_dim + 1] 
        # compute eta
        eta = np.min(self.Y) - params[self.X_dim + 2] # QUESTION: what is this?
        # compute the observed value for g instead of y
        g_ob = np.sqrt(2.0 * (self.Y - eta))

        kernel = GPy.kern.RBF(input_dim=self.X_dim, ARD=True, variance=output_var, lengthscale=l_scales)
        Kng = kernel.K(self.X)
        # QUESTION: does not seem to follow conditional variance form in eqn 6
        
        # compute posterior mean distribution for g TODO update this
        # GPg = GPy.models.GPRegression(self.X, g_ob, kernel, noise_var=1e-8)
        # mg,_ = GPg.predict(self.X)
        mg = g_ob

        # approximate covariance matrix of y using linearisation technique
        Kny = mg * Kng * mg.T + (noise_var+1e-8) * np.eye(Kng.shape[0])

        # compute likelihood terms
        Wi, LW, LWi, W_logdet = pdinv(Kny) # from GPy module
        # Wi = inverse of Kny (ndarray)
        # LW = Cholesky decomposition of Kny (ndarray)
        # LWi = Cholesky decomposition of inverse of Kny (ndarray)
        # W_logdet = log determinant of Kny (float)
        
        alpha, _ = dpotrs(LW, self.Y, lower=1)
        loglikelihood = 0.5 * (-self.Y.size * np.log(2 * np.pi) - self.Y.shape[1] * W_logdet - np.sum(alpha * self.Y))
        # Log marginal likelihood for GP, based on Rasmussen eqn 2.30
        
        return loglikelihood

    def _log_posterior(self, log_params, mean_ln_yminob_minus_eta, var_ln_yminob_minus_eta):
        # Returns posterior, p(y|D, hyperparams, eta)
        # QUESTION: is this true? What is the formula?
        
        params = np.exp(log_params)
        # compute log likelihood
        log_likelihood = self._log_likelihood(log_params)

        # log prior of hyperparameters
        det_l_scales = np.product(params[0:self.X_dim])
        logl_priormu = np.log(0.3 * np.ones(self.X_dim))
        logl_priorvar = np.diag(3.0 * np.ones(self.X_dim))
        prior_l_scales = Gaussianprior(log_params[0: self.X_dim], logl_priormu, logl_priorvar) / det_l_scales
        # Gaussianprior(X, mean, variance)
        # Returns pdf of X based on multivariate Gaussian with given mean and variance
        prior_output_var = Gaussianprior(log_params[self.X_dim], np.log(1.0), 3.0) / params[self.X_dim]
        prior_noise_var = Gaussianprior(log_params[self.X_dim + 1], np.log(0.01), 1.0) / params[self.X_dim + 1]
        prior_ymin_eta = Gaussianprior(log_params[self.X_dim + 2], mean_ln_yminob_minus_eta, var_ln_yminob_minus_eta )/\
                         params[self.X_dim + 2]
        log_posterior = log_likelihood + np.log(prior_l_scales * prior_output_var * prior_noise_var * prior_ymin_eta)
        return log_posterior

    def _samplehyper(self, mean_ln_yminob_minus_eta, var_ln_yminob_minus_eta):
        
        Nsamples = self.burnin + self.mc_samples
        L_d = np.ones(self.X_dim)
        # define initial guess for hyperparameters
        init = np.hstack((np.log(0.3*L_d), np.log(10.0), np.log(1e-2), mean_ln_yminob_minus_eta))

        prior_mu = np.zeros(self.X_dim+3)
        prior_cov = np.diag(1.0 * np.ones(self.X_dim+3))
        prior_cov_chol = sp.linalg.cholesky(prior_cov, lower=True)

        params_array = init
        log_params = np.zeros((Nsamples, len(params_array)))
        sampler_options = {"cur_log_like": None, "angle_range": 0}
        extra_para = (mean_ln_yminob_minus_eta, var_ln_yminob_minus_eta)
        for i in range(Nsamples):
            params_array, current_ll = elliptical_slice(
                params_array,
                self._log_posterior,
                prior_cov_chol,
                prior_mu,
                * extra_para,
                **sampler_options)
            log_params[i,:] = params_array.ravel()
            current_ll = current_ll  # for diagnostics QUESTION: what is this for?
        self.params = np.exp(log_params[self.burnin:, :])

    def _FITBOMM(self,x):
        '''FITBO-Moment Matching acquisition function'''
        x = np.atleast_2d(x)
        noise_param = self.params[:, self.X_dim + 1]
        eta = np.min(self.Y)- self.params[:, self.X_dim + 2]
        N_hypsamples = len(self.params)

        n = x.shape[0]  # number of test point
        Mean_y = np.zeros([N_hypsamples, n])
        Var_y = np.zeros([N_hypsamples, n])
        Entropy_T2 = np.zeros([N_hypsamples, n])
        weights = 1.0 / N_hypsamples

        for i in range(N_hypsamples):

            mg, varg = self.GP[i].predict(x)
            my = 0.5 * (mg ** 2.0) + eta[i]
            vary = (mg ** 2) * varg + noise_param[i]
            Mean_y[i, :] = my[:, 0]
            Var_y[i, :] = vary[:, 0]
            # Compute the 2nd entropy term ( entropy of a gaussian )
            Entropy_T2[i, :] = 0.5 * np.log(2.0 * np.pi * self.e * vary)[:, 0]

        # Compute the 1st entropy term in FITBOMM ( entropy of a gaussian mixture )
        mean_GMM = weights * np.sum(Mean_y, axis=0)
        cov_GMM = weights * np.sum(Var_y + Mean_y ** 2, axis=0) - mean_GMM ** 2
        Entropy_1 = 0.5 * np.log(2.0 * np.pi * self.e * cov_GMM)
        # Compute the acquisition function

        Mutual_info = Entropy_1 - np.mean(Entropy_T2, axis=0)
        # return - Mutual_info for minimiser

        return - Mutual_info

    def _FITBO(self,x):
        '''FITBO-Numerical Quadrature acquisition function'''
        x = np.atleast_2d(x)
        noise_param = self.params[:, self.X_dim + 1]
        eta = np.min(self.Y) - self.params[:, self.X_dim + 2]
        N_hypsamples = len(self.params)

        n = x.shape[0]  # number of test point
        Mean_y = np.zeros([N_hypsamples, n])
        Var_y = np.zeros([N_hypsamples, n])
        Entropy_T2 = np.zeros([N_hypsamples, n])
        weights = 1.0 / N_hypsamples

        for i in range(N_hypsamples):

            mg, varg = self.GP[i].predict(x)
            my = 0.5 * (mg ** 2.0) + eta[i]
            vary = (mg ** 2) * varg + noise_param[i]
            Mean_y[i, :] = my[:, 0]
            Var_y[i, :] = vary[:, 0]
            # Compute the 2nd entropy term ( entropy of a gaussian )
            Entropy_T2[i, :] = 0.5 * np.log(2.0 * np.pi * self.e * vary)[:, 0]

        # Compute the 1st entropy term in FITBOMM ( entropy of a gaussian mixture )
        weights = 1 / N_hypsamples
        Entropy_1 = np.zeros(n)

        for j2 in range(n):
            Entropy_1[j2] = computeEntropyGMM1Dquad(Mean_y[:, j2], Var_y[:, j2], weights, 3)

        # Compute the acquisition function
        Mutual_info = Entropy_1 - np.mean(Entropy_T2, axis=0)
        # return - Mutual_info for minimiser
        return - Mutual_info

    def _fit_GP(self):
        '''collect GPs defined using observed g values and all hyperparameter samples'''
        self.GP = [] # Contains a collection of GPs based on hyperparameter sampling
        self.kernel = [] # Contains a collection of kernels based on hyperparameter sampling
        lscale_param = self.params[:, 0:self.X_dim]
        var_param = self.params[:, self.X_dim]
        noise_param = self.params[:, self.X_dim + 1]
        eta = np.min(self.Y) - self.params[:, self.X_dim + 2] # There are 

        for i in range(len(self.params)): # iterate across each MC sample of hyperparaters (Nsamples = 50)
            diff = self.Y - eta[i]
            diff.clip(min=0) # Change negative values to zero
            self.G = np.sqrt(2.0 * diff)
            self.kernel.append(GPy.kern.RBF(input_dim=self.X_dim, ARD=True, variance=var_param[i], lengthscale=lscale_param[i, :]))
            self.GP.append(GPy.models.GPRegression(self.X, self.G, self.kernel[i], noise_var=(noise_param[i])))

    def _fit_GP_normal(self):
        '''collect GPs defined using observed y values and all hyperparameter samples'''
        self.GP_normal = []
        noise_param = self.params[:, self.X_dim + 1]
        for i in range(len(self.params)):
            self.GP_normal.append(GPy.models.GPRegression(self.X, self.Y, self.kernel[i], noise_var=noise_param[i]))

    def _marginalised_posterior_mean(self, x):
        x = np.atleast_2d(x) # Wrap array x into 2D-array
        n = x.shape[0]
        Meanf = np.zeros([len(self.params), n])
        # dMeanf = np.zeros([len(self.params), n])
        for i in range(len(self.params)):
            my, vary = self.GP_normal[i].predict(x) # GPy.models.predict returns (mean, variance) of prediction
            # dmy, dvary = self.GP_normal[i].predictve_gradients(x)

            Meanf[i, :] = my[:, 0]
            # dMeanf[i, :] = dmy[:, 0]
        pos_mean = np.mean(Meanf, axis=0)
        # dpos_mean = np.mean(dMeanf, axis=0)
        return pos_mean
    
    def _marginalised_posterior_var(self, x):
        '''Marginalize GP-normal over all hyperparam sample'''
        x = np.atleast_2d(x) # Wrap array x into 2D-array
        n = x.shape[0]
        Mean_var = np.zeros([len(self.params), n])
        for i in range(len(self.params)):
            my, vary = self.GP_normal[i].predict(x) # GPy.models.predict returns (mean, variance) of prediction
            Mean_var[i, :] = vary[:, 0]
        pos_var = np.mean(Mean_var, axis=0)
        return pos_var

    def _gloabl_minimser(self,func):
        ntest = 2000;
        Xgrid = np.random.uniform(0.0, 1.0, (ntest, self.X_dim))
        Xtest = np.vstack((Xgrid,self.X))
        Func_value_test = func(Xtest)
        idx_test = np.argmin(Func_value_test)
        X_start = Xtest[idx_test, :]
        # bnds = ((0.0, 1.0), (0.0, 1.0))
        bnds = tuple((li, ui) for li, ui in zip(self.lb, self.ub))
        
        print("bnds")
        print(bnds)
        res = minimize(func, X_start, method='L-BFGS-B', jac=False, bounds=bnds)
        x_opt = res.x[None, :]
        
        # Rounding
        for i in range(len(self.input_type)):
            if self.input_type[i] == True:
                x_opt[0][i] = np.rint(x_opt[0][i])
        
        return x_opt

    def iteration_step_batch(self, num_batches, mc_burn , mc_samples,bo_method, seed, resample_interval, \
                             batch_size = 2, heuristic = "kb", dir_name = 'Exp_Data/'):

        np.random.seed(seed)

        X_optimum = np.atleast_2d(self.arg_opt)
        Y_optimum = np.atleast_2d(np.min(self.Y))
        X_for_L2  = X_optimum
        Y_for_IR  = Y_optimum

        # sample hyperparameters
        self.mc_samples = mc_samples  # number of samples
        self.burnin = mc_burn         # number of burnin
        self.e = np.exp(1)
        # initial guess for log(y_min - eta_hyparameters)
        mean_log_ymin_minus_eta = np.log(1.0)
        var_log_ymin_minus_eta = 0.1
        self._samplehyper(mean_log_ymin_minus_eta, var_log_ymin_minus_eta)
        mean_log_ymin_minus_eta, var_log_ymin_minus_eta = sp.stats.norm.fit(np.log(self.params[:, self.X_dim + 2])) # Fit a normal distribution to sampled etas

        # fit GP models to hyperparameter samples
        self._fit_GP() 
        self._fit_GP_normal()

        # Specify acquisition function
        if bo_method == 'FITBOMM':
            acqu_func = self._FITBOMM
        else:
            acqu_func = self._FITBO
        
        #################### Main changes for batch
        
        batch_X = np.zeros((num_batches, batch_size, self.X_dim))
        batch_Y = np.zeros((num_batches, batch_size, 1))
        
        self.full_PI_value = np.zeros((num_batches, batch_size, 1))
        
        for k in range(num_batches):
            
            # Storing values which will be reset once batch iterations are over
            real_X = copy.deepcopy(self.X)
            real_Y = copy.deepcopy(self.Y)
            
            """ Temporarily not used as deepcopy isnt working
            real_GP = copy.copy(self.GP)
            real_GP_normal = copy.copy(self.GP_normal)
            """
            
            #batch_X = [] # Stores suggested query points for this batch
            
            # Iterate across current batch
            for batch_i in range(batch_size):
                # optimise the acquisition function to get the next query point and evaluate at next query point
                
                x_next = self._gloabl_minimser(acqu_func)
                max_acqu_value = - acqu_func(x_next)
                
                if heuristic == "kb":
                    y_next_guess = batch_proposals.kriging_believer(self, x_next) 
                elif heuristic[0:2] == "cl":
                    cl_setting = heuristic[3:]
                    y_next_guess = batch_proposals.constant_liar(self, x_next, cl_setting = cl_setting)
                elif heuristic == "random":
                    """Fully random, even first sample in batch"""
                    x_next = np.random.rand(1, self.X_dim) # Returns random float 
                    y_next_guess = -10.0 # Arbitary as not used in random batch selector
                elif heuristic == "random_except_1st":
                    """First sample in batch is max of acq function, rest is random"""
                    if batch_i > 0:
                        x_next = np.random.rand(1, self.X_dim) # Returns random float 
                        y_next_guess = 0.5 # Arbitary as not used in random batch selector
                    else:
                        y_next_guess = 0.5 # Arbitary as not used in random batch selector
                        

                # PI Values - calculate before updating GP with guessed values      
                current_y_best = real_Y.min()
                x_next_mean = self._marginalised_posterior_mean(x_next)
                x_next_var = self._marginalised_posterior_var(x_next)
                PI_value = norm.cdf((-(x_next_mean) + current_y_best) / np.sqrt(x_next_var)) 

                self.X = np.vstack((self.X, x_next))
                self.Y = np.vstack((self.Y, y_next_guess)) # Appending Data with guessed values
                
                if heuristic != "random" and heuristic != "random_except_1st": # If random, not necessary to recalculate GP
                    print("Not random, need reset GP")
                    self._fit_GP()
                    self._fit_GP_normal()
                else:
                    print("Random, not resetting GP")
                
                #print("Currently on iteration %d, batch %d" % (k, batch_i))
                
                # Just for recording
                batch_X[k, batch_i, :] = x_next
                batch_Y[k, batch_i, :] = y_next_guess
                self.full_PI_value[k, batch_i, : ] = PI_value 
                
            
            # Resetting back to original real values 
            # TODO: deepcopy doesnt work so we re-initialize GP with original X's every time
            self.X = real_X
            self.Y = real_Y
            self._fit_GP()
            self._fit_GP_normal()    
                        
            # Finding real function values for all query points in batch
            
            cur_batch_X = batch_X[k]

            self.X = np.vstack((self.X, cur_batch_X))
            actual_y = self.func(cur_batch_X) + np.random.normal(0, self.var_noise, (batch_size, 1))
            self.Y = np.vstack((self.Y, actual_y))
                
            #################### Main changes for batch END
            
            # resample hyperparameters
            if k % resample_interval ==0:
                self._samplehyper(mean_log_ymin_minus_eta, var_log_ymin_minus_eta)
                mean_log_ymin_minus_eta, var_log_ymin_minus_eta = sp.stats.norm.fit(
                                                                np.log(self.params[:, self.X_dim + 2]))

            # update the GP models and the minimum observed value
            self._fit_GP()

            # optimise the marginalised posterior mean to get the prediction for the global optimum/optimiser
            self._fit_GP_normal()
            
            x_opt = self._gloabl_minimser(self._marginalised_posterior_mean)
            y_opt = self.func(x_opt)
            X_optimum = np.concatenate((X_optimum, np.atleast_2d(x_opt)))
            Y_optimum = np.concatenate((Y_optimum, np.atleast_2d(y_opt)))
            X_for_L2 = np.concatenate((X_for_L2, np.atleast_2d(X_optimum[np.argmin(Y_optimum),:])))
            Y_for_IR = np.concatenate((Y_for_IR, np.atleast_2d(min(Y_optimum))))

            print("bo:"+ bo_method + ",seed:{seed},itr:{iteration},x_next: {next_query_loc},y_next:{next_query_value}, acq value: {best_acquisition_value},"
                "x_opt:{x_opt_pred},y_opt:{y_opt_pred}"
                .format(seed = seed,
                        iteration=k,
                        next_query_loc=x_next,
                        next_query_value=actual_y,
                        best_acquisition_value=max_acqu_value,
                        x_opt_pred=X_for_L2[-1,:], # QUESTION: why is this always the last value?
                        y_opt_pred=Y_for_IR[-1,:]
                        ))
            
        # Just for saving
        new_dir = dir_name + str(seed) + '_seed/' 
        
        try:
            os.mkdir(new_dir)
        except FileExistsError:
            pass

        file_name = new_dir + heuristic + ',intermediate_vars.pickle'
                
        pickle_dict = {
                "X": self.X, 
                "Y": self.Y, 
                "X_init": self.X_init,
                "Y_init": self.Y_init,
                "PI_values": self.full_PI_value
                }
        
        with open(file_name, 'wb') as f:
            pickle.dump(pickle_dict, f)          

        return X_for_L2, Y_for_IR


# '''
# Test
# '''
# seed_size = 20
# obj_func = branin
# var_noise = 1.0e-3
# d = 2
# sigma0 = np.sqrt(var_noise)
# initialsamplesize = 3

# for j in range(seed_size):
#
#     seed = j
#     np.random.seed(seed)
#     x_ob = np.random.uniform(0., 1., (initialsamplesize, d))
#     y_ob = obj_func(x_ob) + sigma0 * np.random.randn(initialsamplesize, 1)
#
#     bayes_opt = Bayes_opt(obj_func, np.zeros(d), np.ones(d), var_noise)
#     bayes_opt.initialise(x_ob, y_ob)
#     X_query,Y_query,X_optimum,Y_optimum = bayes_opt.iteration_step(iterations=50, mc_burn=500, mc_samples=200, bo_method='FITBOMM',
#                                                                    seed=seed,resample_interval= 5)
#     np.save('branin_results_Xopt', X_optimum)
#     np.save('branin_results_Yopt', Y_optimum)
