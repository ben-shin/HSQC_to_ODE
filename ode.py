import numpy as np
from scipy.integrate import solve_ivp

def dP_dt(t, y, keff, n, kelong, krise, A_func):
    """
    y[0] = P_quenched (Concentration of monomers that are NMR-invisible or quenched)
    y[1] = P_visible  (Concentration of monomers that are NMR-visible)
    """
    Pq, Pv = y
    
    dPq_dt = -krise * Pq
    
    dPv_dt = (krise * Pq) - (keff * Pv**n) - (kelong * Pv * A_func(t))
    
    return [dPq_dt, dPv_dt]

def solve_kinetics(t_span, P0, params, A_func, t_eval=None):
    """
    Parameters:
    P0 - This is the TOTAL monomer concentration.
    params - dict now requires 'krise' and 'f_visible' (fraction already visible at t0)
    """
    f_v = params.get('f_visible', 1.0) # Default to 1.0 (Standard Decay)
    Pv0 = P0 * f_v
    Pq0 = P0 * (1 - f_v)
    
    sol = solve_ivp(
        dP_dt,
        t_span,
        y0=[Pq0, Pv0],
        t_eval=t_eval,
        args=(params['keff'], params['n'], params['kelong'], params['krise'], A_func),
        vectorized=False
    )
    
    return sol
