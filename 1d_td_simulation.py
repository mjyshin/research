# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 13:00:20 2019

@author: matth
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


def mu(x,t,epsilon,alpha,beta):
    return 1/epsilon*(x-x**3+alpha+beta*np.cos(2*np.pi*t))

def dW(dt):
    return np.random.normal(scale=np.sqrt(dt))

def simulate(x0,t0,epsilon,alpha,beta,sigma,numsims):
    # Deterministic simulation
    N = 5000
    ti = t0-3.0
    tf = t0+3.0
    dt = float(tf-t0)/N
    T = np.arange(t0,tf,dt)
    T2 = np.arange(ti,tf,dt)
    A0 = -1.0
    B0 = 1.0
    As = integrate.odeint(mu,A0,T2,args=(epsilon,alpha,beta)).reshape(T2.shape)
    Bs = integrate.odeint(mu,B0,T2,args=(epsilon,alpha,beta)).reshape(T2.shape)
    As = As[len(As)//2:]
    Bs = Bs[len(Bs)//2:]
    #plt.figure()
    #plt.plot(T,As,T,Bs)
    
    # Stochastic simulation
    X = np.zeros(N)
    X[0] = x0
    numA = 0
    numB = 0
    for _ in range(numsims):
        for i in range(1,T.size):
            t = T[i-1]
            x = X[i-1]
            if x<=As[i-1]:
                numA = numA+1
                for j in range(i,T.size):
                    X[j] = np.nan
                break
            elif x>=Bs[i-1]:
                numB = numB+1
                for j in range(i,T.size):
                    X[j] = np.nan
                break
            X[i] = x+mu(x,t,epsilon,alpha,beta)*dt+sigma*dW(dt)
        #plt.plot(T,X)
    qhat = numB/(numA+numB)
    return qhat


# Set up problem
epsilon = 0.25
alpha = 0.1
beta = 0.7
sigma = 0.4
Nt = 50
Nx = 50

# Run Euler-Maruyama
np.random.seed(0)
ts = np.linspace(-2.5,2.5,2*Nt)[Nt:]
xs = np.linspace(-1.5,1.5,Nt)
numsims = 10
qhats = np.zeros((len(xs),len(ts)))
for i,x in enumerate(xs):
    for j,t in enumerate(ts):
        qhats[i,j] = simulate(x,t,epsilon,alpha,beta,sigma,numsims)

# Plot committor estimate
t0 = ts[0]
tf = ts[-1]
xmin = xs[0]
xmax = xs[-1]
plt.figure()
im = plt.imshow(qhats,origin='lower',extent=[t0,tf,xmin,xmax],aspect='auto',interpolation='none',cmap=plt.cm.seismic)
plt.xlabel(r'$t$')
plt.ylabel(r'$x$')
cbar = plt.colorbar(im)
cbar.set_label(r'$\hat{q}\! ^+\! (x,t)$')

# Save simulation data
saveqhats = 'data/tdep_committor/e%.02f_a%.02f_b%.02f_s%.02f_%dx%d/qhats' % (epsilon,alpha,beta,sigma,Nt,Nx)
np.save(saveqhats.replace('.', 'p'), qhats)

plt.show()
