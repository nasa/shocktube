#! /usr/bin/env python
# -*- coding:utf-8 -*-


#import os, sys
import numpy as np
from numpy import *
import matplotlib.pyplot as pyplot

from matplotlib import rc
rc('font', family='serif')
rc('lines', linewidth=1.5)
rc('font', size=14)



#def func_cons2prim(q,gamma):
#    # Primitive variables
#    r=q[0];
#    u=q[1]/r;
#    E=q[2]/r;
#    p=(gamma-1.)*r*(E-0.5*u**2);
#
#    return (r,u,p)

#def func_prim2cons(r,u,p,gamma):
#    # Conservative variables
#    q0=r;
#    q1=r*u;
#    q2=p/(gamma-1.)+0.5*r*u**2;
#    q =np.array([ q0, q1, q2 ]);
#
#    return (q)
    
def func_flux(q,gamma):
    # Primitive variables
    r=q[0]
    u=q[1]/r
    E=q[2]/r
    p=(gamma-1.)*r*(E-0.5*u**2)
    
    # Flux vector
    F0 = np.array(r*u)
    F1 = np.array(r*u**2+p)
    F2 = np.array(u*(r*E+p))
    flux=np.array([ F0, F1, F2 ])
    
    return (flux)

def flux_roe(q,dx,gamma,a,nx):

    # Compute primitive variables and enthalpy
    r=q[0]
    u=q[1]/r
    E=q[2]/r
    p=(gamma-1.)*r*(E-0.5*u**2)
    htot = gamma/(gamma-1)*p/r+0.5*u**2
    
    # Initialize Roe flux
    Phi=np.zeros((3,nx-1))
    
    for j in range (0,nx-1):
    
        # Compute Roe averages
        R=sqrt(r[j+1]/r[j]);                          # R_{j+1/2}
        rmoy=R*r[j];                                  # {hat rho}_{j+1/2}
        umoy=(R*u[j+1]+u[j])/(R+1);                   # {hat U}_{j+1/2}
        hmoy=(R*htot[j+1]+htot[j])/(R+1);             # {hat H}_{j+1/2}
        amoy=sqrt((gamma-1.0)*(hmoy-0.5*umoy*umoy));  # {hat a}_{j+1/2}
        
        # Auxiliary variables used to compute P_{j+1/2}^{-1}
        alph1=(gamma-1)*umoy*umoy/(2*amoy*amoy)
        alph2=(gamma-1)/(amoy*amoy)

        # Compute vector (W_{j+1}-W_j)
        wdif = q[:,j+1]-q[:,j]
        
        # Compute matrix P^{-1}_{j+1/2}
        Pinv = np.array([[0.5*(alph1+umoy/amoy), -0.5*(alph2*umoy+1/amoy),  alph2/2],
                        [1-alph1,                alph2*umoy,                -alph2 ],
                        [0.5*(alph1-umoy/amoy),  -0.5*(alph2*umoy-1/amoy),  alph2/2]]);
                
        # Compute matrix P_{j+1/2}
        P    = np.array([[ 1,              1,              1              ],
                        [umoy-amoy,        umoy,           umoy+amoy      ],
                        [hmoy-amoy*umoy,   0.5*umoy*umoy,  hmoy+amoy*umoy ]]);
        
        # Compute matrix Lambda_{j+1/2}
        lamb = np.array([[ abs(umoy-amoy),  0,              0                 ],
                        [0,                 abs(umoy),      0                 ],
                        [0,                 0,              abs(umoy+amoy)    ]]);
                      
        # Compute Roe matrix |A_{j+1/2}|
        A=np.dot(P,lamb)
        A=np.dot(A,Pinv)
        
        # Compute |A_{j+1/2}| (W_{j+1}-W_j)
        Phi[:,j]=np.dot(A,wdif)
        
    #==============================================================
    # Compute Phi=(F(W_{j+1}+F(W_j))/2-|A_{j+1/2}| (W_{j+1}-W_j)/2
    #==============================================================
    F = func_flux(q,gamma);
    Phi=0.5*(F[:,0:nx-1]+F[:,1:nx])-0.5*Phi
    
    dF = (Phi[:,1:-1]-Phi[:,0:-2])
    
    return (dF)

def buildIC(pointCount, numCells):
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
        timeEnd = 0.20
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

    return p_vector, u_vector, r_vector, timeEnd


#Fill out the constants and inputs

#Constants
COURANT_NUM    = 0.50               # Courant Number - CFL
IC = 1 # 6 IC cases are available

# Inputs
specificHeatsRatio  = 1.4                # Ratio of specific heats - gamma
numCells = 400                # Number of cells - ncells
x_lower =0.; x_upper = 1.       # Limits of computational domain -start and final
step = (x_upper-x_lower)/numCells   # Step size - dx
pointCount = numCells+1               # Number of points - nx
x_domain = np.linspace(x_lower+step/2.,x_upper,pointCount) # Mesh - x

#populate numpy arrays
p_vector, u_vector, r_vector, timeEnd = buildIC(pointCount, numCells)

#Calculate values with newly populated vectors
E0 = p_vector/((specificHeatsRatio-1.)*r_vector)+0.5*u_vector**2 # Total Energy density
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
    pyplot.plot(x_domain, p_vector, 'b-')
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
    
    # plt.show()

# Loop from 0 to timeEnd
tCur  = 0
itCount = 0
deltaTime=COURANT_NUM*step/max(abs(u_vector)+a)         # Using the system's largest eigenvalue - dt

while tCur < timeEnd:

    q0 = q.copy()
    dF = flux_roe(q0,step,specificHeatsRatio,a,pointCount)
    
    q[:,1:-2] = q0[:,1:-2]-deltaTime/step*dF
    q[:,0]=q0[:,0]; q[:,-1]=q0[:,-1]; # Dirichlet BCs
    
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
        
    # Using pyplot plot
    if itCount%2 == 0:
        fig,axes = pyplot.subplots(nrows=4, ncols=1, num=1, figsize=(10, 8), clear=True)
        fig.suptitle('Roe Scheme')

        pyplot.subplot(4, 1, 1)
        #pyplot.title('Roe scheme')
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
        #pyplot.show()
        import os
        os.makedirs('roe_scheme_results',exist_ok=True)
        fig.savefig(f"roe_scheme_results/fig_Sod_Roe_it_{itCount:04d}.png", dpi=300)
