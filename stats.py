import numpy as np

def fit_statistics(P_exp, P_model):
  """
  Returns fitting stats
  RMSE and MAE
  """
  residuals = P_exp - P_model

  rmse = np.sqrt(np.mean(residuals**2))
  mae = np.mean(np.abs(residuals))

  return {
    "RMSE": rmse,
    "MAE": mae
  }
