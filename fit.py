import numpy as np
from ode import solve_kinetics

def residuals(theta, t, P_exp, P0, A_func):
  """
  Returns residuals for least squares fitting
  theta = [keff, n, kelong]
  """
  keff, n, kelong = theta
  params = {"keff": keff, "n": n, "kelong": kelong}

  sol = solve_kinetics(
    t_span=(t[0], t[-1]),
    P0=P0,
    params=params,
    A_func=A_func,
    t_eval=t
  )

  if not sol.success:
    return 1e6 * np.ones_like(P_exp)

  P_model = sol.y[0]
  return P_exp - P_model
