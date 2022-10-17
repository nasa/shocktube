#! /usr/bin/env python
# -*- coding:utf-8 -*-

################################################################
#
# Class 06: 1-D Euler system of equations
# Solve equation using the Lax-Wendroff scheme
# Author: P. S. Volpiani
# Date: 01/08/2020
#
################################################################

import os, sys
import numpy as np
import matplotlib.pyplot as pyplot

from numpy import *
from matplotlib import rc

rc('font', family='serif')
rc('lines', linewidth=1.5)
rc('font', size=14)
#pyplot.rc('legend',**{'fontsize':11})

######################################################################
#             Solving 1-D Euler system of equations
#                 using the Lax-Wendroff scheme
#
#             dq_i/dt + df_i/dx = 0, for x \in [a,b]
#
# This code solves the Sod's shock tube problem (IC=1)
#
# t=0                                 t=tEnd
# Density                             Density
#   ****************|                 *********\
#                   |                           \
#                   |                            \
#                   |                             ****|
#                   |                                 |
#                   |                                 ****|
#                   ***************                       ***********
#
# Domain cells (I{i}) reference:
#
#                |           |   u(i)    |           |
#                |  u(i-1)   |___________|           |
#                |___________|           |   u(i+1)  |
#                |           |           |___________|
#             ...|-----0-----|-----0-----|-----0-----|...
#                |    i-1    |     i     |    i+1    |
#                |-         +|-         +|-         +|
#              i-3/2       i-1/2       i+1/2       i+3/2
#
######################################################################


#Constants
COURANT_NUM    = 0.50               # Courant Number - CFL
IC = 1 # 6 IC cases are available

# Parameters
specificHeatsRatio  = 1.4                # Ratio of specific heats - gamma
numCells = 400                # Number of cells - ncells
x_lower =0.; x_upper = 1.       # Limits of computational domain - start and final bounds
step = (x_upper-x_lower)/numCells   # Step size - dx
pointCount = numCells+1               # Number of points - nx
x_domain = np.linspace(x_lower+step/2.,x_upper,pointCount) # Mesh

# Build IC
r_vector = np.zeros(pointCount)
u_vector = np.zeros(pointCount)
p_vector = np.zeros(pointCount)
splitCells = int(numCells/2)

if IC == 1:
    print ("Configuration 1, Sod's Problem")
    p_vector[:splitCells] = 1.0  ; p_vector[splitCells:] = 0.1
    u_vector[:splitCells] = 0.0  ; u_vector[splitCells:] = 0.0
    r_vector[:splitCells] = 1.0  ; r_vector[splitCells:] = 0.125
    timeEnd = 0.20; #tEnd
elif IC== 2:
    print ("Configuration 2, Left Expansion and right strong shock")
    p_vector[:splitCells] = 1000.; p_vector[splitCells:] = 0.1
    u_vector[:splitCells] = 0.0  ; u_vector[splitCells:] = 0.0
    r_vector[:splitCells] = 3.0  ; r_vector[splitCells:] = 0.2
    timeEnd = 0.01
elif IC == 3:
    print ("Configuration 3, Right Expansion and left strong shock")
    p_vector[:splitCells] = 7.   ; p_vector[splitCells:] = 10.
    u_vector[:splitCells] = 0.0  ; u_vector[splitCells:] = 0.0
    r_vector[:splitCells] = 1.0  ; r_vector[splitCells:] = 1.0
    timeEnd = 0.10
elif IC == 4:
    print ("Configuration 4, Shocktube problem of G.A. Sod, JCP 27:1, 1978")
    p_vector[:splitCells] = 1.0  ; p_vector[splitCells:] = 0.1
    u_vector[:splitCells] = 0.75 ; u_vector[splitCells:] = 0.0
    r_vector[:splitCells] = 1.0  ; r_vector[splitCells:] = 0.125
    timeEnd = 0.17
elif IC == 5:
    print ("Configuration 5, Lax test case: M. Arora and P.L. Roe: JCP 132:3-11, 1997")
    p_vector[:splitCells] = 3.528; p_vector[splitCells:] = 0.571
    u_vector[:splitCells] = 0.698; u_vector[splitCells:] = 0.0
    r_vector[:splitCells] = 0.445; r_vector[splitCells:] = 0.5
    timeEnd = 0.15
elif IC == 6:
    print ("Configuration 6, Mach = 3 test case: M. Arora and P.L. Roe: JCP 132:3-11, 1997")
    p_vector[:splitCells] = 10.33; p_vector[splitCells:] = 1.0
    u_vector[:splitCells] = 0.92 ; u_vector[splitCells:] = 3.55
    r_vector[:splitCells] = 3.857; r_vector[splitCells:] = 1.0
    timeEnd = 0.09

E0 = p_vector/((specificHeatsRatio-1.)*r_vector)+0.5*u_vector**2 # Total Energy density - E0
a = sqrt(specificHeatsRatio*p_vector/r_vector)            # Speed of sound
q  = np.array([r_vector,r_vector*u_vector,r_vector*E0])   # Vector of conserved variables

if (False):
    fig = pyplot.subplots()
    ax1 = pyplot.subplot(4, 1, 1)
    #pyplot.title('Lax-Wendroff scheme')
    pyplot.plot(x_domain, r_vector, 'k-')
    pyplot.ylabel('$rho$',fontsize=18)
    pyplot.tick_params(axis='x',bottom=False,labelbottom=False)
    pyplot.grid(True)
    
    ax2 = pyplot.subplot(4, 1, 2)
    pyplot.plot(x_domain, u_vector, 'r-')
    pyplot.ylabel('$U$',fontsize=18)
    pyplot.tick_params(axis='x',bottom=False,labelbottom=False)
    pyplot.grid(True)

    ax3 = pyplot.subplot(4, 1, 3)
    pyplot.plot(x_domain, p0, 'b-')
    pyplot.ylabel('$P$',fontsize=18)
    pyplot.tick_params(axis='x',bottom=False,labelbottom=False)
    pyplot.grid(True)
    
    ax4 = pyplot.subplot(4, 1, 4)
    pyplot.plot(x_domain, E0, 'g-')
    pyplot.ylabel('$E$',fontsize=18)
    pyplot.grid(True)
    pyplot.xlim(x_lower,x_upper)
    pyplot.xlabel('x',fontsize=18)
    pyplot.subplots_adjust(left=0.2)
    pyplot.subplots_adjust(bottom=0.15)
    pyplot.subplots_adjust(top=0.95)
    
    pyplot.show()

# Solver loop
tCur  = 0 #t
itCount = 0
deltaTime=COURANT_NUM*step/max(abs(u_vector)+a)  # Using the system's largest eigenvalue - dt

while tCur < timeEnd:
    
    # I. Predictor step
    # =================
    q0 = q.copy()
    
    # Primary variables
    r=q0[0]
    u=q0[1]/r
    E=q0[2]/r
    p=(specificHeatsRatio-1.)*r*(E-0.5*u**2)
    
    # Flux vector of conserved properties
    F0 = np.array(r*u)
    F1 = np.array(r*u**2+p)
    F2 = np.array(u*(r*E+p))
    flux=np.array([ F0, F1, F2 ])
    
    qm  = np.roll(q0, 1)
    qp  = np.roll(q0,-1)
    fm  = np.roll(flux, 1)
    fp  = np.roll(flux,-1)

    qpHalf = (qp+q0)/2. - deltaTime/(2.*step)*(fp-flux)
    qmHalf = (qm+q0)/2. - deltaTime/(2.*step)*(-fm+flux)
    
    # II. Corrector step
    # ==================
        
    r=qpHalf[0]
    u=qpHalf[1]/r
    E=qpHalf[2]/r
    p=(specificHeatsRatio-1.)*r*(E-0.5*u**2)
    F0 = np.array(r*u)
    F1 = np.array(r*u**2+p)
    F2 = np.array(u*(r*E+p))
    FqpHalf=np.array([ F0, F1, F2 ])
    
    r=qmHalf[0]
    u=qmHalf[1]/r
    E=qmHalf[2]/r
    p=(specificHeatsRatio-1.)*r*(E-0.5*u**2)
    F0 = np.array(r*u)
    F1 = np.array(r*u**2+p)
    F2 = np.array(u*(r*E+p))
    FqmHalf=np.array([ F0, F1, F2 ])
    
    dF = FqpHalf - FqmHalf
    
    q = q0-deltaTime/step*dF
    q[:,0]=q0[:,0]
    q[:,-1]=q0[:,-1] # BCs
   
  
    # Compute primary variables
    rho=q[0]
    u=q[1]/rho
    E=q[2]/rho
    p=(specificHeatsRatio-1.)*rho*(E-0.5*u**2)
    a=sqrt(specificHeatsRatio*p/rho)
    if min(p)<0: print ('negative pressure found!')
    
    # Update/correct time step
    deltaTime=COURANT_NUM*step/max(abs(u)+a)
    
    # Update time and iteration counter
    tCur=tCur+deltaTime
    itCount+=1
    
    # Plot solution
    if itCount%2 == 0:
        print (itCount)
        fig,axes = pyplot.subplots(nrows=4, ncols=1)
        pyplot.subplot(4, 1, 1)
        #pyplot.title('Lax-Wendroff scheme')
        pyplot.plot(x_domain, rho, 'k-')
        pyplot.ylabel('$rho$',fontsize=16)
        pyplot.tick_params(axis='x',bottom=False,labelbottom=False)
        pyplot.grid(True)

        pyplot.subplot(4, 1, 2)
        pyplot.plot(x_domain, u, 'r-')
        pyplot.ylabel('$U$',fontsize=16)
        pyplot.tick_params(axis='x',bottom=False,labelbottom=False)
        pyplot.grid(True)

        pyplot.subplot(4, 1, 3)
        pyplot.plot(x_domain, p, 'b-')
        pyplot.ylabel('$p$',fontsize=16)
        pyplot.tick_params(axis='x',bottom=False,labelbottom=False)
        pyplot.grid(True)
    
        pyplot.subplot(4, 1, 4)
        pyplot.plot(x_domain, E, 'g-')
        pyplot.ylabel('$E$',fontsize=16)
        pyplot.grid(True)
        pyplot.xlim(x_lower,x_upper)
        pyplot.xlabel('x',fontsize=16)
        pyplot.subplots_adjust(left=0.2)
        pyplot.subplots_adjust(bottom=0.15)
        pyplot.subplots_adjust(top=0.95)
        #  pyplot.show()
        fig.savefig("analytical_plots/fig_Sod_LW_it"+str(itCount)+".png", dpi=300)


