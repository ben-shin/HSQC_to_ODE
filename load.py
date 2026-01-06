import numpy as np

def load_csv(filepath, delimiter=",", skip_header=0):
  """
  Loads a csv with
  column 0 - time (seconds)
  column 1 - I/I0

  Returns
  t - numpy array (time)
  I_ratio - numpy array (I/I0)
  """
  data = np.loadtxt(filepath, delimiter=delimiter, skiprows=skip_header)

  t = data[:, 0]
  I_ratio = data[:, 1]

  return t, I_ratio
