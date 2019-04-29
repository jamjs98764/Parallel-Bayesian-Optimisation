#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:45:16 2017

@author: robin
"""

import numpy as np
import math
import sys

'''MCMC sampling methods'''

def slice_sweep(xx, logdist, widths=1.0, step_out=True, Lp=None):
    """simple axis-aligned implementation of slice sampling for vectors

         xx_next = slice_sample(xx, logdist)
         samples = slice_sample(xx, logdist, N=200, burn=20)

     Inputs:
                xx  D,  initial state (or array with D elements)
           logdist  fn  function: log of unnormalized probability of xx
            widths  D,  or 1x1, step sizes for slice sampling (default 1.0)
          step_out bool set to True (default) if widths sometimes far too small
                Lp  1,  Optional: logdist(xx) if have already evaluated it

     Outputs:
                xx  D,  final state (same shape as at start)
     If Lp was provided as an input, then return tuple with second element:
                Lp  1,  final log-prob, logdist(xx)
    """
    # Iain Murray 2004, 2009, 2010, 2013, 2016
    # Algorithm orginally by Radford Neal, e.g., Annals of Statistic (2003)
    # See also pseudo-code in David MacKay's text book p375

    # startup stuff
    D = xx.size
    widths = np.array(widths)
    if widths.size == 1:
        widths = np.tile(widths, D)
    output_Lp = False
    if Lp is None:
        log_Px = logdist(xx)
    else:
        log_Px = Lp
    perm = np.array(range(D))

    # Force xx into vector for ease of use:
    xx_shape = xx.shape
    logdist_vec = lambda x: logdist(np.reshape(x, xx_shape))
    xx = xx.ravel().copy()
    x_l = xx.copy()
    x_r = xx.copy()
    xprime = xx.copy()

    # Random scan through axes
    np.random.shuffle(perm)
    for dd in perm:
        log_uprime = log_Px + np.log(np.random.rand())
        # Create a horizontal interval (x_l, x_r) enclosing xx
        rr = np.random.rand()
        x_l[dd] = xx[dd] - rr*widths[dd]
        x_r[dd] = xx[dd] + (1-rr)*widths[dd]
        if step_out:
            # Typo in early book editions: said compare to u, should be u'
            while logdist_vec(x_l) > log_uprime:
                x_l[dd] = x_l[dd] - widths[dd]
            while logdist_vec(x_r) > log_uprime:
                x_r[dd] = x_r[dd] + widths[dd]

        # Inner loop:
        # Propose xprimes and shrink interval until good one found
        while True:
            xprime[dd] = np.random.rand()*(x_r[dd] - x_l[dd]) + x_l[dd]
            log_Px = logdist_vec(xprime)
            if log_Px > log_uprime:
                break # this is the only way to leave the while loop
            else:
                # Shrink in
                if xprime[dd] > xx[dd]:
                    x_r[dd] = xprime[dd]
                elif xprime[dd] < xx[dd]:
                    x_l[dd] = xprime[dd]
                else:
                    raise Exception('BUG DETECTED: Shrunk to current '
                        + 'position and still not acceptable.')
        xx[dd] = xprime[dd]
        x_l[dd] = xprime[dd]
        x_r[dd] = xprime[dd]

    if output_Lp:
        return xx, log_Px
    else:
        return xx

def slice_sample(logposterfunc, init, iters, burnin, sigma, step_out=True):
    """
    Slice sampler based on http://homepages.inf.ed.ac.uk/imurray2/teaching/09mlss/
    """
    # init  - initial guess for the parameters
    # iters - number of samples generated
    # burnin - number of burnin
    # sigma - step-size parameter
    # set up empty sample holder
    D = len(init)
    rawsamples = np.zeros((D, iters))

    # initialize
    xx = init.copy()

    for i in range(iters):
        perm = list(range(D))
        np.random.shuffle(perm)
        last_llh = logposterfunc(xx)

        for d in perm:
            llh0 = last_llh + np.log(np.random.rand())
            rr = np.random.rand(1)
            x_l = xx.copy()
            x_l[d] = x_l[d] - rr * sigma[d]
            x_r = xx.copy()
            x_r[d] = x_r[d] + (1 - rr) * sigma[d]

            if step_out:
                llh_l = logposterfunc(x_l)
                while llh_l > llh0:
                    x_l[d] = x_l[d] - sigma[d]
                    llh_l = logposterfunc(x_l)

                llh_r = logposterfunc(x_r)
                while llh_r > llh0:
                    x_r[d] = x_r[d] + sigma[d]
                    llh_r = logposterfunc(x_r)

            x_cur = xx.copy()
            while True:
                xd = np.random.rand() * (x_r[d] - x_l[d]) + x_l[d]
                x_cur[d] = xd.copy()
                last_llh = logposterfunc(x_cur)
                if last_llh > llh0:
                    xx[d] = xd.copy()
                    break
                elif xd > xx[d]:
                    x_r[d] = xd
                elif xd < xx[d]:
                    x_l[d] = xd
                else:
                    raise RuntimeError('Slice sampler shrank too far.')

        if i % 1000 == 0: print('iteration', i)

        rawsamples[:, i] = xx.copy().ravel()

    samples = rawsamples[:, burnin:]
    return samples

def elliptical_slice(xx, log_like_fn, prior_chol, prior_mean, *log_like_args, **sampler_args):
    ''' Elliptical slice sampler '''
    cur_log_like = sampler_args.get('cur_log_like', None)
    angle_range = sampler_args.get('angle_range', 0)

    if cur_log_like is None:
        cur_log_like = log_like_fn(xx, *log_like_args)

    if np.isneginf(cur_log_like):
        raise Exception("Elliptical Slice Sampler: initial logprob is -inf for inputs %s" % xx)
    if np.isnan(cur_log_like):
        raise Exception("Elliptical Slice Sampler: initial logprob is NaN for inputs %s" % xx)

    nu = np.dot(prior_chol,
                np.random.randn(xx.shape[0]))  # don't bother adding mean here, would just subtract it at update step
    hh = np.log(np.random.rand()) + cur_log_like
    # log likelihood threshold -- LESS THAN THE INITIAL LOG LIKELIHOOD

    # Set up a bracket of angles and pick a first proposal.
    # "phi = (theta'-theta)" is a change in angle.
    if angle_range <= 0:
        # Bracket whole ellipse with both edges at first proposed point
        phi = np.random.rand() * 2 * math.pi
        phi_min = phi - 2 * math.pi
        phi_max = phi
    else:
        # Randomly center bracket on current point
        phi_min = -angle_range * np.random.rand();
        phi_max = phi_min + angle_range;
        phi = np.random.rand() * (phi_max - phi_min) + phi_min;

    # Slice sampling loop
    while True:
        # Compute xx for proposed angle difference
        # and check if it's on the slice
        xx_prop = (xx - prior_mean) * np.cos(phi) + nu * np.sin(phi) + prior_mean

        cur_log_like = log_like_fn(xx_prop, *log_like_args)

        if cur_log_like > hh:
            # New point is on slice, ** EXIT LOOP **
            return xx_prop, cur_log_like

        # Shrink slice to rejected point
        if phi > 0:
            phi_max = phi
        elif phi < 0:
            phi_min = phi
        else:
            sys.stderr.write('Initial x: %s\n' % xx)
            # sys.stderr.write('initial log like = %f\n' % initial_log_like)
            sys.stderr.write('Proposed x: %s\n' % xx_prop)
            sys.stderr.write('ESS log lik = %f\n' % cur_log_like)
            raise Exception('BUG DETECTED: Shrunk to current position '
                            'and still not acceptable.');

        # Propose new angle difference
        phi = np.random.rand() * (phi_max - phi_min) + phi_min
