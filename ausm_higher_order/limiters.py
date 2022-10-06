import numpy as np 

def minmod(r:float) -> float:
    """minmod limiter (Roe, 1986)

    References:
        https://en.wikipedia.org/wiki/Flux_limiter

    Args:
        r (float): ratio of successive gradients of state variables r[i] = u[i] - u[i-1]

    Returns:
        float: limiter
    """
    
    return np.maximum( 0, np.minimum(1,r) )

def superbee(r:float) -> float:
    """Superbee limiter (Roe, 1986)

    References:
        https://en.wikipedia.org/wiki/Flux_limiter

    Args:
        r (float): _description_

    Returns:
        float: limiter
    """
    return np.maximum(0,np.minimum(2*r,1),np.minimum(r,2))