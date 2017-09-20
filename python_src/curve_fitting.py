import numpy as np
import scipy as sp
import numpy.linalg as la
import pandas as pd
import scipy.optimize as spo
import scipy.interpolate as spi
import scipy.sparse as sparse
import scipy.signal as signal
import cPickle as pickle
import time
import matplotlib.pyplot as plt

def overlapMat(m, t, k):
    
    if k == 3:
    
        B1B1 = np.zeros([m+2, m+2], float)
        B1B1[k:m,k:m] += 1.0/3.0 * np.diag(t[k+1:m+1] - t[k:m])
        B1B1[k-1:m-1,k-1:m-1] += 1.0/3.0 * np.diag(t[k+1:m+1] - t[k:m])
        B1B1[k-1:m,k-1:m] += 1.0/6.0*np.diag(t[k+1:m+1] - t[k:m], -1)
        B1B1[k-1:m,k-1:m] += 1.0/6.0*np.diag(t[k+1:m+1] - t[k:m], 1)

        A = np.zeros([3, m], float)

        hh = (t[3:m+3] - t[0:m])*(t[2:m+2] - t[0:m])
        A[0, hh.nonzero()] = 6.0 / hh[hh.nonzero()]
        hh = (t[3:m+3] - t[0:m])*(t[3:m+3] - t[1:m+1])
        A[1, hh.nonzero()] += -6.0 / hh[hh.nonzero()]
        hh = (t[4:m+4] - t[1:m+1])*(t[3:m+3] - t[1:m+1])
        A[1, hh.nonzero()] += -6.0 / hh[hh.nonzero()]
        hh = (t[4:m+4] - t[1:m+1])*(t[4:m+4] - t[2:m+2])
        A[2, hh.nonzero()] = 6.0 / hh[hh.nonzero()]

        Sigma = np.zeros([m,m], float)
        for i in range(m):
            for j in range(i, m):
                Sigma[i, j] = A[:, i].dot(B1B1[i:i+3, j:j+3].dot(A[:, j]))
                Sigma[j, i] = Sigma[i, j]
                
    else:
        
        Sigma = np.zeros([m,m], float)
    
        for i in range(m):
            ci = np.zeros(m, float)
            ci[i] = 1.0
            for j in range(i, np.min([i+k+1, m])):
                cj = np.zeros(m, float)
                cj[j] = 1.0

                integrand = lambda x: spi.BSpline(t, ci, k).derivative(2)(x) * spi.BSpline(t, cj, k).derivative(2)(x)
                Sigma[i, j] = spint.quad(integrand, t[k+1], t[n - k - 1])[0]
                Sigma[j, i] = Sigma[i, j]
        
                
    sSigma = sparse.csc_matrix(Sigma)
                
    return sSigma

def evalBasis(m, t, k):
    
    Bitj = np.zeros([m, len(t)-2*k], float)
    for i in range(m):
        c = np.zeros(m, float)
        c[i] = 1.0
        spline = spi.BSpline(t, c, k)
        Bitj[i] = spline(t[k:len(t)-k])

#     sX = sparse.csc_matrix(Bitj.transpose())
        
    return Bitj.transpose()                   

# def CV_obj(lam, x, y, dist_param, penalty, penalty_grad, k, t, c0):
        
#     cv_obj = 0.0

# #     fig, ax = plt.subplots(1,1)
#     for i in range(len(x)):

#         x0 = np.delete(x, i)
#         y0 = np.delete(y, i)
#         dist_param0 = np.delete(dist_param, i)
        
#         ix = np.arange(len(x))
#         ix0 = np.delete(ix, i)
        
#         args=(ix0, y0, dist_param0, lam)
#         c = min_penalty(penalty, penalty_grad, c0, args)
        
#         fx  = spi.BSpline(t, c, k)(x[i])

#         n = dist_param[i]
#         p = 1.0 * y[i] / n
        
#         cv_obj += n * (np.log(1+np.exp(fx)) - p*fx)
                
# #         ax.plot(np.linspace(x[0], x[-1], 1000), spline(np.linspace(x[0], x[-1], 1000)), '-', label="{}".format(i))
        
# #     ax.plot(x, sp.special.erfinv(1- 2.0 *1.0 *y / dist_param), 'o')
# #     ax.legend()
# # #     ax.set_ylim(-0.1, 1.1)
# #     plt.show()

#     cv_obj /= len(x) 


#     return cv_obj

def GCV_approx_obj(lam, x, y, dist_param, penalty, penalty_grad, k, t, c0, Sigma, X):
        
        
    gamma = np.exp(lam)
    
#     print lam
    
    args=(y, dist_param, gamma)
    c = min_penalty(penalty, penalty_grad, c0, args)
        
    fx = X.dot(c)
        
    mu = dist_param * sp.special.expit(fx)
    D = dist_param * sp.special.expit(fx)**2.0 * np.exp(-fx)
    DhalfX = np.diag(np.sqrt(D)).dot(X)
    
    # factor of 2 * lambda * M might be off
    
    M = DhalfX.transpose().dot(DhalfX) + 2.0 * gamma * Sigma
    
    Minv = la.inv(M)
    
    A = DhalfX.dot(Minv).dot(DhalfX.transpose())
        
#     if show:
#         print 1.0 * y / dist_param
#         print 1.0/D * (y - mu)**2.0
            
    cv_obj = np.sum(1.0/D * (y - mu)**2.0) / len(y) / (1.0 - np.trace(A) / len(y))**2.0

    # cv_obj = np.sum(1.0/D * (y - mu)**2.0 / (1.0 - np.diag(A))**2.0) / len(y)
        
    return cv_obj


def min_penalty(penalty, penalty_grad, c0, args):
    res = spo.minimize(penalty, c0, jac=penalty_grad, args=args,
           method='BFGS', options={'gtol': np.sqrt(np.finfo(float).eps)})
    
    # print res
    
    return res.x


# def min_CV(x0, args):
#     alpha = 0.1
#     beta1 = 0.9
#     beta2 = 0.999
#     eps = 1e-4  
    
#     x = x0
#     m = 0.0
#     v = 0.0
    
#     for t in range(1000):
#         g = (CV_obj(x+eps, *args) - CV_obj(x-eps, *args)) / (2.0*eps)
        
#         print t, x, CV_obj(x, *args), g
#         if np.abs(g) < eps:
#             return x
                
#         m = beta1 * m + (1.0-beta1) * g
#         v = beta2 * v + (1.0-beta2) * g*g
#         mhat = m / (1.0-beta1**(t+1))
#         vhat = v / (1.0-beta2**(t+1))
#         x -= alpha * mhat / (np.sqrt(vhat) + eps)
    
#     return x
    
def min_CV(lam0, args):

    res = spo.minimize(CV_obj, [lam0], jac=None, args=args,
                       method='L-BFGS-B', options={'ftol': np.sqrt(np.finfo(float).eps), 
                                           'gtol': 1e-4, 'eps':1e-4}, bounds=[(1e-4, None)])

    print res
    
    return res.x


def min_GCV(args, bounds):

    res = spo.minimize_scalar(GCV_approx_obj,bounds=bounds, args=args, method='bounded', tol=1e-4)

    # print res
    
    return res.x


def lossBinom(y, n, fx):
    
    p = 1.0 * y / n
    eps = np.finfo(float).eps
    
    return np.sum( n * (np.log(1.0 + np.exp(fx)) - p*fx) ) / len(y)
    
def lossBinom_grad(y, n, fx):
    
    p = 1.0 * y / n
    eps = np.finfo(float).eps
    
    return n * (sp.special.expit(fx) - p ) / len(y)
    
def lossGauss(y, sigma, fx):
    
    return np.sum((y - fx)**2 / sigma**2) / len(y)
   

def curveFitSpline(x, y, dist_param, dist="gauss", lam0=0.0, CV=True, plot=False, verbose=False): 
    
    k = 3
        
    n = 2*k + len(x)
    
    m = n - (k+1)
    
        
    t = np.zeros(n, float)
    t[0:k] = x[0] 
    t[n-k-1:] = x[-1]
    t[k:n-k] = x
    
    Sigma = overlapMat(m, t, k)
    X = evalBasis(m, t, k)

    def obj(c, y, dist_param):
        fx = X.dot(c)
    
        return lossBinom(y, dist_param, fx)
    
    def obj_grad(c, y, dist_param):
        fx = X.dot(c)
        
        g = lossBinom_grad(y, dist_param, fx) 
        
        return X.transpose().dot(g)
        
        
    smooth = lambda c: c.dot(Sigma.dot(c))
    penalty = lambda c,  y, dist_param, lam: obj(c, y, dist_param) + lam * smooth(c)
    
    smooth_grad = lambda c: 2.0 * Sigma.dot(c)
    
    penalty_grad = lambda c,  y, dist_param, lam: obj_grad(c, y, dist_param) + lam * smooth_grad(c)
       
    c0 = np.zeros(m, float)
    
    args=(y, dist_param, np.exp(lam0))  
    
    # print "check grad", spo.check_grad(penalty, penalty_grad, c0, *args) 
    
    
    c0 = min_penalty(penalty, penalty_grad, np.copy(c0), args)
       
   
    
        
    if CV:
        
         
        
        lam_list = np.linspace(0, 20, 32)
        cv_list = []
        for i in lam_list:
    #         cv_list.append(CV_obj(i, x, y, dist_param, penalty, penalty_grad, k, t, c0))
            cv_list.append(GCV_approx_obj(i, x, y, dist_param, penalty, penalty_grad, k, t, c0, Sigma, X))
         
        cv_list = np.array(cv_list)
        
        local_min = signal.argrelmin(cv_list)[0]   
        local_min = np.concatenate([local_min, [len(lam_list)-1]])
        
        lam0 = lam_list[local_min][0]
        
        # lam0 = lam_list[local_min][np.argmin(cv_list[local_min])]

        # lam0 = lam_list[np.argmin(cv_list)]
        bounds = (lam0-1.0, lam0+1.0)
                
#         args = (x, y, dist_param, penalty, penalty_grad, k, t, c0)
#         lam = min_CV(lam0, args)

        args = (x, y, dist_param, penalty, penalty_grad, k, t, c0, Sigma, X)
        lam = min_GCV(args, bounds)        
        
        if plot:
            fig, ax = plt.subplots(1,1) 
            ax.plot(lam_list, cv_list)
            ax.scatter([lam], [GCV_approx_obj(lam, x, y, dist_param, penalty, penalty_grad, k, t, c0, Sigma, X)], marker='o', color='k')
            ax.scatter(lam_list[local_min], cv_list[local_min], marker='^', color='g')
            ax.set_yscale('log')
            plt.show()

        
        args=(y, dist_param, np.exp(lam))  
        c = min_penalty(penalty, penalty_grad, c0, args)
        
    else:
        lam = lam0
        c = c0


   
        
    
    
#     cv_list = []
#     for i in lam_list:
#         cv_list.append(CV_obj(np.exp(i), x, y, dist_param, penalty, penalty_grad, k, t, c0))
# #         cv_list.append(GCV_approx_obj(i, x, y, dist_param, penalty, penalty_grad, k, t, c0, M, Bitj))
#     ax.plot(lam_list, cv_list)
    

               
        
        
#     print "gcv", GCV_approx_obj(lam0, x, y, dist_param, penalty, penalty_grad, k, t, c0, M, Bitj)
    
    
    if verbose:
    
        print "Lambda:", lam, np.exp(lam)

        print "Residuals", obj(c0, y, dist_param), obj(c, y, dist_param)
        print "Smoothness", smooth(c0), smooth(c)
        
    spline = spi.BSpline(t, c, k)
        
    return spline

def calcTransition(spline, a, b):
    
#     fcenter = sp.special.logit(0.5)
#     flow = sp.special.logit(0.25)
#     fup = sp.special.logit(0.75)
    
    fcen = 0.0
    flow = np.log(3.0)
    fup = -np.log(3.0)
    
    func = lambda x: spline(x) - fcen
    xcen = spo.brentq(func, a, b)
    
    func = lambda x: spline(x) - flow
    
    if func(a) < 0.0:
        xlow = a
    else:
        xlow = spo.brentq(func, a, b)
    
    func = lambda x: spline(x) - fup
    if func(b) > 0.0:
        xup = b
    else:
        xup = spo.brentq(func, a, b)

    width = xup - xlow
        
    return xcen, width


    

    
    
    
# ###############################################3

def wilson_up(p, n):
    z = 1.0
    return 1.0 / (1.0 + z**2/n) * (p + 0.5*z**2 / n + z*np.sqrt(1.0/n * p * (1.0 - p) + 0.25*z**2/n**2)) - p

def wilson_low(p, n):
    z = 1.0
    return p - 1.0 / (1.0 + z**2/n) * (p + 0.5*z**2 / n - z*np.sqrt(1.0/n * p * (1.0 - p) + 0.25*z**2/n**2))


# def binomMLE(x, y, dist_param, func, p0=None, bounds=None):
    
#     n = dist_param.astype(np.int)
#     k = (n*y).astype(np.int)
        
#     fit_func = lambda theta: -np.sum(k * np.log(func(x, *theta) + np.sqrt(np.finfo(float).eps)) 
#                                           + (n - k) * np.log(1.0 - func(x, *theta) + np.sqrt(np.finfo(float).eps)))
    
#     res = spo.minimize(fit_func, p0, jac=None,
#                        method='L-BFGS-B', options={'gtol': np.sqrt(np.finfo(float).eps), 'disp': False},
#                       bounds=bounds)
        
#     return res.x

# def mcError(x, y, dist_param, fit_func, p0, fit_jac=None, dist="gaussian", bounds=None, nresample=100, plot=False):
    
#     p = np.zeros([len(p0), nresample], float)
    
#     x = np.array(x, float)
#     y = np.array(y, float)
#     dist_param = np.array(dist_param, float)
        
#     for n in range(nresample):
# #         if n % 100 == 0:
# #             print n
#         if dist=="binomial":

#             y_tmp = np.array(rand.binomial(dist_param.astype(int), y) / dist_param)

#             avg =  binomMLE(x, y_tmp, dist_param, fit_func, p0=p0, bounds=bounds)

#             p[:, n] = avg
#         else:
#             sigma = np.array(dist_param)
#             sigma[sigma <= 0.0] = 1.0
#             y_tmp = np.array(rand.normal(loc=y, scale=sigma))

#             popt, pcov = spo.curve_fit(fit_func, x, y_tmp, p0=p0, sigma=sigma, absolute_sigma=True, maxfev=10000)
            
#             p[:, n] = popt
    
#     if plot:
    
#         fig, axes = plt.subplots(1, len(p0), figsize=(24, 8))
#         for i in range(len(p0)):
#             sns.distplot(p[i], ax=axes[i], rug=True, rug_kws={"color": "g"},
#                          kde_kws={"color": "k", "lw": 3},
#                          hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "g"})
    
#     return np.mean(p, axis=1), np.std(p, axis=1), sp.stats.skew(p, axis=1)



# def lossBinom(y, n, fx):
    
#     p = 1.0 * y / n
#     eps = np.finfo(float).eps
    
#     fx0 = np.clip(fx, eps, 1-eps)
        
#     return -np.sum( n * ( p * np.log(fx0) + (1-p) * np.log(1.0 - fx0) ) )
    
# def lossGauss(y, sigma, fx):
    
#     return np.sum((y - fx)**2 / sigma**2)
                  
                  
# def curveFitFunc(x, y, dist_param, func, p0=None, bounds=None, dist="gauss"):
    
#     if dist=="gauss":
#         obj = lambda theta: lossGauss(y, dist_param, func(x, *theta))
#     else:
#         obj = lambda theta: lossBinom(y, dist_param, func(x, *theta))
    
#     res = spo.minimize(func, p0, jac=None,
#                        method='L-BFGS-B', options={'gtol': np.sqrt(np.finfo(float).eps), 'disp': False},
#                       bounds=bounds)
    
#     return res.x

# def mcResample(x, y, dist_param, fit_func, p0, dist="gauss", bounds=None, nresample=100, plot=False):
    
#     p = np.zeros([len(p0), nresample], float)
    
#     x = np.array(x, float)
#     y = np.array(y, float)
#     dist_param = np.array(dist_param, float)
        
#     for n in range(nresample):
# #         if n % 100 == 0:
# #             print n
#         if dist=="guass":
#             sigma = np.array(dist_param)
#             sigma[sigma <= 0.0] = 1.0
#             y_tmp = np.array(rand.normal(loc=y, scale=sigma))

#             popt = curveFitFunc(x, y_tmp, dist_param, fit_func, dist="gauss", p0=p0, bounds=bounds)

#             p[:, n] = popt
        
#         else:
#             n = np.array(dist_param)
#             y_tmp = np.array(rand.binomial(n.astype(int), y) / n)
            
#             popt = curveFitFunc(x, y_tmp, dist_param, fit_func, dist="binom", p0=p0, bounds=bounds)

#             p[:, n] = popt
           
    
#     if plot:
    
#         fig, axes = plt.subplots(1, len(p0), figsize=(24, 8))
#         for i in range(len(p0)):
#             sns.distplot(p[i], ax=axes[i], rug=True, rug_kws={"color": "g"},
#                          kde_kws={"color": "k", "lw": 3},
#                          hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "g"})
    
#     return np.mean(p, axis=1), np.std(p, axis=1), sp.stats.skew(p, axis=1)
