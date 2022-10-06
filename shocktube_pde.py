import sys
import numpy as np
import matplotlib.pyplot as plt 
import torch
from nangs import PDE
import math 

class shocktube_pde(PDE):
    def __init__(self,inputs,outputs, gamma:float=1.4, CFL:float=0.5):
        """Initializes Burger's PDE 

        Args:
            nu (float, optional): [description]. Defaults to 0.1.
        """
        super().__init__(inputs,outputs)
        self.gamma = torch.as_tensor(gamma)
        self.CFL = CFL

    def computePDELoss(self,inputs:torch.Tensor,outputs:torch.Tensor):
        """Compute the loss in burger's equation 

        Args:
            inputs (torch.Tensor): x, t as tensors with shape of (npoints, 2)
            outputs (torch.Tensor): this is p, u, and r as tensors with shape of (npoints,3)
        """
        # To compute du_dx, du_dy we have to extract u and v from the outputs and use them in the gradients
        p, u, rho = outputs[:,0], outputs[:,1], outputs[:,2]

        E = p/((self.gamma-1.0)*rho)+0.5*u**2
        a = torch.sqrt(self.gamma*p/rho)            # Speed of sound
        # q  = torch.stack([rho,rho*u,rho*E],dim=1)         # Vector of conserved variables
        # F =  torch.stack([rho*u, rho*u*u + p, u*(rho*E + p)],dim=1)

        # We want the output to be u and the input to be x,y,z this computes the gradient for du_dx, du_dy, du_dt
        # This is q 
        dQ1_dt = self.computeGrads(rho, inputs)[:,0] # output, input
        dQ2_dt = self.computeGrads(rho*u, inputs)[:,0] # output, input
        dQ3_dt = self.computeGrads(rho*E, inputs)[:,0] # output, input

        dF1_dx = self.computeGrads(rho*u, inputs)[:,1] # output, input
        dF2_dx = self.computeGrads(rho*u*u, inputs)[:,1]
        dF3_dx = self.computeGrads(u*(rho*E + p), inputs)[:,1]

        # Burgers PDE
        return { 
            'pde-1': dQ1_dt + dF1_dx,
            'pde-2': dQ2_dt + dF2_dx,
            'pde-3': dQ3_dt + dF3_dx,
            }