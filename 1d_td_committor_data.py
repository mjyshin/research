# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 18:26:27 2019

@author: matth
"""

import numpy as np
from scipy import integrate


def mu(x,t,epsilon,alpha,beta):
    return 1/epsilon*(x-x**3+alpha+beta*np.cos(2*np.pi*t))

def mu_rev(x,t,epsilon,alpha,beta):
    return -1/epsilon*(x-x**3+alpha+beta*np.cos(2*np.pi*(-t)))

def muz(z,t,epsilon,alpha,beta,A,B,Adot,Bdot):
    D = B-A
    Ddot = Bdot-Adot
    x = D*z+A
    mux = mu(x,t,epsilon,alpha,beta)
    #return (mux+Adot)/D + (A-x)*Ddot/D**2
    return D*(mux-Adot-z*Ddot)

def f(z,t,epsilon,alpha,beta,sigma,A,B,Adot,Bdot):
    #sigmaz = sigma/(B-A)
    Mz = integrate.quad(muz,0,z,args=(t,epsilon,alpha,beta,A,B,Adot,Bdot))[0]
    return np.exp(-2/sigma**2*Mz)

def r(z,t,epsilon,alpha,beta,sigma,A,B,Adot,Bdot):
    if z<=0:
        return 0.0
    elif z>=1:
        return 1.0
    
    Fz = integrate.quad(f,0,z,args=(t,epsilon,alpha,beta,sigma,A,B,Adot,Bdot))[0]
    F = integrate.quad(f,0,1,args=(t,epsilon,alpha,beta,sigma,A,B,Adot,Bdot))[0]
    return Fz/F


# Set up problem
epsilon = 0.25
alpha = 0.1
beta = 0.7
sigma = 0.4
Nt = 100
Nx = 100
ts = np.linspace(-2.5,2.5,2*Nt)
ts_rev = np.linspace(-5.0,5.0,4*Nt)[:2*Nt]
xs = np.linspace(-1.5,1.5,Nx)


# Set up attractors and separatrix
A0 = -1.0
B0 = 1.0
C0 = 0.0
As = integrate.odeint(mu,A0,ts,args=(epsilon,alpha,beta)).reshape(ts.shape)
Bs = integrate.odeint(mu,B0,ts,args=(epsilon,alpha,beta)).reshape(ts.shape)
Cs = integrate.odeint(mu_rev,C0,ts_rev,args=(epsilon,alpha,beta)).reshape(ts.shape)
ts = ts[Nt:]
As = As[Nt:]
Bs = Bs[Nt:]
Cs = Cs[Nt:][::-1]
Adots = mu(As,ts,epsilon,alpha,beta).reshape(ts.shape)
Bdots = mu(Bs,ts,epsilon,alpha,beta).reshape(ts.shape)

'''
# Plot A and B
plt.figure()
plt.plot(ts,As)
plt.plot(ts,Bs)

# Plot Adot and Bdot
plt.figure()
plt.plot(ts,Adots)
plt.plot(ts,Bdots)
'''

# Compute committor
zs = np.zeros([Nx,Nt])
rs = np.zeros(zs.shape)
for i,x in enumerate(xs):
    for j,t in enumerate(ts):
        zs[i,j] = (x-As[j])/(Bs[j]-As[j])
        rs[i,j] = r(zs[i,j],t,epsilon,alpha,beta,sigma,As[j],Bs[j],Adots[j],Bdots[j])

# Save committor data
savets = 'data/tdep_committor/e%.02f_a%.02f_b%.02f_s%.02f_%dx%d/ts' % (epsilon,alpha,beta,sigma,Nt,Nx)
np.save(savets.replace('.', 'p'), ts)
savexs = 'data/tdep_committor/e%.02f_a%.02f_b%.02f_s%.02f_%dx%d/xs' % (epsilon,alpha,beta,sigma,Nt,Nx)
np.save(savexs.replace('.', 'p'), xs)
saveAs = 'data/tdep_committor/e%.02f_a%.02f_b%.02f_s%.02f_%dx%d/As' % (epsilon,alpha,beta,sigma,Nt,Nx)
np.save(saveAs.replace('.', 'p'), As)
saveBs = 'data/tdep_committor/e%.02f_a%.02f_b%.02f_s%.02f_%dx%d/Bs' % (epsilon,alpha,beta,sigma,Nt,Nx)
np.save(saveBs.replace('.', 'p'), Bs)
saveCs = 'data/tdep_committor/e%.02f_a%.02f_b%.02f_s%.02f_%dx%d/Cs' % (epsilon,alpha,beta,sigma,Nt,Nx)
np.save(saveCs.replace('.', 'p'), Cs)
saveAdots = 'data/tdep_committor/e%.02f_a%.02f_b%.02f_s%.02f_%dx%d/Adots' % (epsilon,alpha,beta,sigma,Nt,Nx)
np.save(saveAdots.replace('.', 'p'), Adots)
saveBdots = 'data/tdep_committor/e%.02f_a%.02f_b%.02f_s%.02f_%dx%d/Bdots' % (epsilon,alpha,beta,sigma,Nt,Nx)
np.save(saveBdots.replace('.', 'p'), Bdots)
savezs = 'data/tdep_committor/e%.02f_a%.02f_b%.02f_s%.02f_%dx%d/zs' % (epsilon,alpha,beta,sigma,Nt,Nx)
np.save(savezs.replace('.', 'p'), zs)
savers = 'data/tdep_committor/e%.02f_a%.02f_b%.02f_s%.02f_%dx%d/rs' % (epsilon,alpha,beta,sigma,Nt,Nx)
np.save(savers.replace('.', 'p'), rs)
