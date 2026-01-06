import numpy as np
from scipy.integrate import solve_ivp

def dP_dt(t, P, keff, n, kelong, A_func):
  """
  Parameters:
  t - float, time in seconds
  P - float, [P]
  keff - fload
  n - float
  kelong - float
  A_func - callable, A(t)

  Returns:
  dP/dt
  """
  return -keff * P**n - kelong * P * A_func(t)

def solve_kinetics(t_span, P0, params, A_func, t_eval=None):
  """
  Solve the ODE
  Parameters:
  t_span - tuple, (t_start, t_end)
  P0 - float, [P]init
  params - dict, {'keff':... etc}
  A_func - callable, A(t)
  t_eval - array-like, time points to evaluate solution

  Returns:
  solution - OdeResult
  """
  sol = solve_ivp(
    dP_dt,
    t_span,
    y0=[P0],
    t_eval=t_eval,
    args=(params['keff'], params['n'], params['kelong'], A_func),
    vectorized=False
  )

  return sol
