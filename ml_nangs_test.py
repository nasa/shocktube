'''
    ml_test.py
    This file reads the simulation model and predicts the value of u and v at random x and y points for a given time and compares it to the analytical solution
'''
import torch, sys
sys.path.insert(0,'../../nangs')
from nangs import MLP 
import os.path as osp 
import json 
import numpy as np 
from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation, rc
from pathlib import Path
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def compute_results(model:torch.nn.Module,x:np.ndarray,t:float):
    """[summary]

    Args:
        model (torch.nn.Module): model 
        X (np.ndarray): vector containing x coordinates
        t (float): time in seconds

    Returns:
        (tuple): containing

            **p** (np.ndarray): pressure 
            **u** (np.ndarray): v velocity
            **rho** (np.ndarray): v velocity
    """
    # Create the inputs 
    
    t = x*0+t 
    x = torch.tensor(x,dtype=torch.float32).to(device)    
    t = torch.tensor(t,dtype=torch.float32).to(device)
    input = torch.stack((t,x),dim=1)
    out = model(input)
    out=out.cpu().numpy()
    return out[:,0], out[:,1],out[:,2]
    
def plot_results(x:np.ndarray,p:np.ndarray,u:np.ndarray,rho:np.ndarray,t:float):
    """_summary_

    Args:
        p (np.ndarray): _description_
        u (np.ndarray): _description_
        rho (np.ndarray): _description_
        t (float): _description_
    """
    gamma = 1.4 
    E = p/((gamma-1.0)*rho)+0.5*u**2
    fig,axes = plt.subplots(nrows=4, ncols=1)
    plt.subplot(4, 1, 1)
    #plt.title('Lax-Wendroff scheme')
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
    # plt.xlim(x_ini,x_fin)
    plt.xlabel('x',fontsize=16)
    plt.subplots_adjust(left=0.2)
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(top=0.95)
    plt.savefig(f'ml_plots/shocktube_t={t:0.4f}.png')

if __name__=="__main__":
    

    assert osp.exists('settings.json'), "Need the settings file that defines the setup conditions"

    ''' 
        Load the settings files
    '''
    with open('settings.json','r') as f:
        settings = json.load(f)    
        config = [c for c in settings['Configurations'] if c['id'] == settings['Configuration_to_run']][0]
        ncells = settings['ncells']
        x_ini =0.0; x_fin = settings['xmax']       # Limits of computational domain
        half = (x_fin + x_ini)/2.0
        
        dx = (x_fin-x_ini)/ncells   # Step size
        nx = ncells+1               # Number of points
        x = np.linspace(0,x_fin,nx) # Mesh
        t = np.arange(0,config['tmax'],0.001) # user will change this 

    config_name = config['name']
    if osp.exists(f'{config_name}.pt'):
        data = torch.load(f'{config_name}.pt')

    model = MLP(data['num_inputs'],data['num_outputs'],data['n_layers'],data['neurons'])
    model.load_state_dict(data["model"])
    model.to(device)

    p = Path("ml_plots/")
    p.mkdir(parents=True, exist_ok=True)

    print('Evaluating and saving data')
    p_history = list()
    u_history = list()
    rho_history = list()
    for i in trange(len(t)):
        p,u,rho = compute_results(model,x,t[i])
        p_history.append(copy.deepcopy(p))
        u_history.append(copy.deepcopy(u))
        rho_history.append(copy.deepcopy(rho))

    print('Creating figures')
    for i in trange(len(t)):
        plot_results(x,p_history[i],u_history[i],rho_history[i],t[i])
    
