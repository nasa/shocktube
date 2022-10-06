import os
import numpy as np
import json 
import matplotlib.pyplot as plt
from ausm import flux_ausm_plus
from matplotlib import rc
from pathlib import Path
rc('font', family='serif')
rc('lines', linewidth=1.5)
rc('font', size=14)
#plt.rc('legend',**{'fontsize':11})


def update_euler(q:np.ndarray,F:np.ndarray,dx:float,dt:float,ng:int) -> np.ndarray:
    """Updates the solution

    Args:
        q (np.ndarray): [3,nx] [rho, rhou, rhoE]
        F (np.ndarray): Differencing scheme applied to flux vector
        dx (float): grid spacing in x direction
        dt (float): time increment 
        ng (int): number of ghost cells

    Returns:
        np.ndarray: qnew, updated q vector 
    """
    ncells = q.shape[1]
    for i in range(ng,ncells-2):
        q[:,i] = q[:,i] - dt/dx * (F[:,i]-F[:,i-1]) # dx already factored into the differentiation, use forward difference

    q[:,0] = q[:,1]     # Dirichlet BCs
    q[:,-1] = q[:,-2]
    return q


with open('settings.json','r') as f:
    settings = json.load(f)
    config = [c for c in settings['Configurations'] if c['id'] == settings['Configuration_to_run']][0]

# Parameters
CFL    = config['CFL']               # Courant Number
gamma  = config['gamma']             # Ratio of specific heats
ncells = settings['ncells']          # Number of cells
x_ini =0.; x_fin = 1.       # Limits of computational domain
dx = (x_fin-x_ini)/ncells   # Step size
nghost_cells = 1            # Number of ghost cells on each boundary
x = np.arange(x_ini-dx*nghost_cells, x_fin+2*dx*nghost_cells, dx) # Mesh
nx = len(x)               # Number of points
# Build IC
r0 = np.zeros(nx)
u0 = np.zeros(nx)
p0 = np.zeros(nx)
halfcells = int(nx/2)

p0[:halfcells] = config['left']['p0']; p0[halfcells:] = config['right']['p0'] 
u0[:halfcells] = config['left']['u0']; u0[halfcells:] = config['right']['u0']
r0[:halfcells] = config['left']['r0']; r0[halfcells:] = config['right']['r0']
tEnd = config['tmax']

E0 = p0/((gamma-1.)*r0)+0.5*u0**2 # Total Energy density
a0 = np.sqrt(gamma*p0/r0)            # Speed of sound
q  = np.array([r0,r0*u0,r0*E0])   # Vector of conserved variables

# Solver loop
t  = 0
it = 0
a  = a0
dt=CFL*dx/max(abs(u0)+a0)         # Using the system's largest eigenvalue

while t < tEnd:
    q0 = q.copy()
    F_half = flux_ausm_plus(q0,gamma) # Calculates the flux at every 1/2 point

    q = update_euler(q0,F_half,dx,dt,nghost_cells)
    # q = update_RK4(q,dt,dx,nghost_cells,gamma)
    # Compute primary variables
    rho=q[0,:]
    u=q[1,:]/rho
    E=q[2,:]/rho
    p=(gamma-1.)*rho*(E-0.5*u**2)
    a=np.sqrt(gamma*p/rho)
    if min(p)<0: 
        print ('negative pressure found!')
    
    # Update/correct time step
    dt=CFL*dx/max(abs(u)+a)
    
    # Update time and iteration counter
    t=t+dt; it=it+1
      
    # Plot solution
    if it%2 == 0:
        fig,axes = plt.subplots(nrows=4, ncols=1, num=1, figsize=(10, 8), clear=True)
        fig.suptitle('AUSM+ Scheme')

        plt.subplot(4, 1, 1)
        #plt.title('Roe scheme')
        plt.plot(x, rho, 'k-')
        plt.ylabel('$rho$',fontsize=16)
        plt.tick_params(axis='x',bottom=False,labelbottom=False)
        plt.grid(True)

        plt.subplot(4, 1, 2)
        plt.plot(x, u, 'r-')
        plt.ylabel('$U$',fontsize=16)
        plt.tick_params(axis='x',bottom=False,labelbottom=False)
        plt.grid(True)

        plt.subplot(4, 1, 3)
        plt.plot(x, p, 'b-')
        plt.ylabel('$p$',fontsize=16)
        plt.tick_params(axis='x',bottom=False,labelbottom=False)
        plt.grid(True)
    
        plt.subplot(4, 1, 4)
        plt.plot(x, E, 'g-')
        plt.ylabel('$E$',fontsize=16)
        plt.grid(True)
        plt.xlim(x_ini,x_fin)
        plt.xlabel('x',fontsize=16)
        plt.subplots_adjust(left=0.2)
        plt.subplots_adjust(bottom=0.15)
        plt.subplots_adjust(top=0.95)
        #plt.show()
        os.makedirs('ausm+_results',exist_ok=True) 
        fig.savefig(f"ausm+_results/fig_Sod_AUSM+_plus_it_{it:04d}.png", dpi=300)
