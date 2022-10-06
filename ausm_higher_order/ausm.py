import numpy as np 
import math 


def M_plus_minus(M:float):
    """Equation 19a ausm+ part I

    Args:
        M (float): Mach Number 

    Returns:
        _type_: _description_
    """
    beta = 0.125
    if np.abs(M) >= 1:
        M_plus = 0.5*(M+np.abs(M)) # Eqn 19a
        M_minus = 0.5*(M-np.abs(M))
    else:
        M_plus = 0.25*(M+1)**2 + beta * (M*M-1)**2  # Eqn 19b, M_beta
        M_minus = -0.25*(M-1)**2 - beta * (M*M-1)**2
    return M_plus, M_minus

def P_alpha_plus_minus(M:float):
    """Equation 21b ausm+ Part I

    Args:
        M (float): Mach 
    """
    alpha = 0.1875 # 3/16
    if (np.abs(M) >= 1):
        P_plus = 0.5*(1+np.sign(M)) 
        P_minus = 0.5*(1-np.sign(M))
    else:
        P_plus = 0.25*(M+1)**2 * (2-M)+alpha*M*(M*M-1)**2
        P_minus = 0.25*(M-1)**2 * (2+M)-alpha*M*(M*M-1)**2
        
    return P_plus, P_minus

def flux_ausm_plus(q:np.ndarray,gamma:float):
    """AUSM+ Flux Splitting Scheme
    
    References:
        Liou, M. S. (1996). A sequel to ausm: Ausm+. Journal of computational Physics, 129(2), 364-382.

    Args:
        q (np.ndarray): [3,nx] [rho, rhou, rhoE] State
        gamma (float): ratio of specific heats 

    Returns:
        np.ndarray: Flux
    """
    # AUSM Mach Number 
    r = q[0,:]
    u = q[1,:]/r
    E = q[2,:]/r  # E = T*Cv # Kenji says maybe use Temperature instead of Energy then convert to Energy 

    P=(gamma-1.)*r*(E-0.5*u**2)
    a = np.sqrt(gamma*P/r)      
    M = u/a                     # Computes mach number at each location on the grid
    H = E+P/r
    F_half = np.zeros((3,q.shape[1]-1)) # rho*u, rho*u*u+P, rho*u*H
    F_L = np.zeros((3,))
    F_R = np.zeros((3,))
    P_half = np.zeros((3,))    
    for i in range(q.shape[1]-1):
        a_star = a[i]
        a_tilda_L = a_star**2 / np.max([a_star,np.abs(u[i])]) 
        a_tilda_R = a_star**2 / np.max([a_star,np.abs(u[i+1])]) 
        a_half = np.min([a_tilda_L, a_tilda_R])

        M_plus,_ = M_plus_minus(u[i]/a_half)
        _,M_minus = M_plus_minus(u[i+1]/a_half)
        m_half = M_plus + M_minus  # Eqn 16 Interface mach number: M+(j) + M-(j+1) Note: this was derived from Eqn 13

        m_half_plus = 0.5 *(m_half + np.abs(m_half))
        m_half_minus = 0.5 *(m_half - np.abs(m_half))

        P_alpha_plus, _ = P_alpha_plus_minus(M[i])
        _, P_alpha_minus = P_alpha_plus_minus(M[i+1])

        p_half = P_alpha_plus * P[i] + P_alpha_minus*P[i+1]  # Eqn 20b         


        F_L[0] = r[i]
        F_L[1] = r[i]*u[i] 
        F_L[2] = r[i]*H[i]

        F_R[0] = r[i+1]
        F_R[1] = r[i+1]*u[i+1] 
        F_R[2] = r[i+1]*H[i+1]
        
        P_half[1] = p_half
        F_half[:,i] = a_half * (m_half_plus * F_L + m_half_minus * F_R) + P_half # A1-A3 in AUSM+ paper 1 summary. 
    
    return F_half

