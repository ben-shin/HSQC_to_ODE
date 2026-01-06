import numpy as np

def i_to_conc(I_ratio, P0):
  """
  Convert I/I0 to [P]

  Parameters:
  I_ratio - numpy array
  P0 - float, initial concentration at t0

  Returns:
  P - numpy array, conc/time
  """
  return P0 * I_ratio
