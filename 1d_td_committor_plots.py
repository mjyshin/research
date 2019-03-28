# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 18:24:36 2019

@author: matth
"""

import numpy as np
import matplotlib.pyplot as plt


# Select data
epsilon = 0.25
alpha = 0.1
beta = 0.7
sigma = 0.4
Nt = 100
Nx = 100

# Load committor data
folder = ('data/tdep_committor/e%.02f_a%.02f_b%.02f_s%.02f_%dx%d/' % (epsilon,alpha,beta,sigma,Nt,Nx)).replace('.', 'p')
ts = np.load(folder+'ts.npy')
xs = np.load(folder+'xs.npy')
As = np.load(folder+'As.npy')
Bs = np.load(folder+'Bs.npy')
Cs = np.load(folder+'Cs.npy')
Adots = np.load(folder+'Adots.npy')
Bdots = np.load(folder+'Bdots.npy')
zs = np.load(folder+'zs.npy')
rs = np.load(folder+'rs.npy')

# Plot committor
t0 = ts[0]
tf = ts[-1]
xmin = xs[0]
xmax = xs[-1]
plt.figure()
im = plt.imshow(rs,origin='lower',extent=[t0,tf,xmin,xmax],aspect='auto',interpolation='none',cmap=plt.cm.seismic)
plt.plot(ts,As,color='b')
plt.plot(ts,Bs,color='r')
plt.plot(ts,Cs,color='w',linestyle='--')
plt.xlabel(r'$t$')
plt.ylabel(r'$x$')
cbar = plt.colorbar(im)
cbar.set_label(r'$q\! ^+\! (x,t)$')

# Compute and plot nullclines
T = np.linspace(ts[0],ts[-1],10000)
nclines = np.zeros([3,len(T)])
for j,t in enumerate(T):
    [nclines[0,j], nclines[1,j], nclines[2,j]] = np.sort(np.roots([-1,0,1,alpha+beta*np.cos(2*np.pi*t)]))
    if nclines[0,j]==nclines[1,j]:
        nclines[0,j] = np.nan
        nclines[1,j] = np.nan
    if nclines[1,j]==nclines[2,j]:
        nclines[1,j] = np.nan
        nclines[2,j] = np.nan
plt.plot(T,nclines[0,:],color='c',linewidth=1) # Lower nullcline
plt.plot(T,nclines[1,:],color='w',linewidth=1,linestyle=':')
plt.plot(T,nclines[2,:],color='m',linewidth=1) # Upper nullcline

# Save figure
committorfig = ('figures/tdep_committor/e%.02f_a%.02f_b%.02f_s%.02f_%dx%d' % (epsilon,alpha,beta,sigma,Nt,Nx)).replace('.', 'p')
plt.savefig(committorfig+'.png', format='png', dpi=1200)

'''
# Plot and save committor simulation
qhats = np.load(folder+'qhats.npy')
plt.figure()
im = plt.imshow(qhats,origin='lower',extent=[t0,tf,xmin,xmax],aspect='auto',interpolation='none',cmap=plt.cm.seismic)
plt.plot(ts,As,color='b')
plt.plot(ts,Bs,color='r')
plt.xlabel(r'$t$')
plt.ylabel(r'$x$')
cbar = plt.colorbar(im)
cbar.set_label(r'$\hat{q}\! ^+\! (x,t)$')

simulationfig = ('figures/tdep_committor/sim_e%.02f_a%.02f_b%.02f_s%.02f_%dx%d' % (epsilon,alpha,beta,sigma,Nt,Nx)).replace('.', 'p')
plt.savefig(simulationfig+'.png', format='png', dpi=1200)
'''

plt.show()
