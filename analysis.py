import argparse
import numpy as np
from scipy.optimize import least_squares
from load import load_csv
from i_to_conc import i_to_conc
from ode import dP_dt
from ode import solve_kinetics
from stats import fit_statistics
from fit import residuals

def main():
  parser = argparse.ArgumentParser(
    description="Kinetic analysis of I/I0 timeseries"
  )

  parser.add_argument(
    "--data",
    type=str,
    required=True,
    help="Path to csv file"
  )

  parser.add_argument(
    "--p0",
    type=float,
    required=True,
    help="Initial concentration [P]"
  )

  parser.add_argument(
    "--n",
    type=float,
    default=2.0,
    help="Reaction order"
  )

  parser.add_argument(
    "--kelong",
    type=float,
    default=0.05,
    help="Elongation rate"
  )

  args = parser.parse_args()

  # load data
  t, I_ratio = load_csv(args.data)
  
  # convert to conc
  p_exp = i_to_conc(I_ratio, args.p0)

  # initial guesses for fitting
  theta0 = [1e-3, args.n, args.kelong]
  
  # define A(t)
  def A_func(t):
    return 1.0 # just a placeholder

  best_rmse = n.inf
  best_result = None
  best_n = None

  # loop over allowed integer n
  for n_candidate in range(args.nmin, args.nmax + 1):
    # initial guess for keff and kelongs only
    theta0 = [1e-3, args.kelong]

    # residual function with fixed n
    def residuals_fixed(theta, t=t, P_exp=P_exp, P0=args.p0, n=n_candidate):
      keff, kelong = theta
      return residuals([keff, n, kelong], t, P_exp, P0, A_func)

    #fit keff and kelong
    result = least_squares(
      residuals_fixed,
      theta0,
      bounds=([0, 0], [np.inf, np.inf])
    )

    # compute RMSE
    keff_fit, kelong_fit = result.x
    params_fit = {"keff": keff_fit, "n": n_candidate "kelong": kelong_fit}
    sol = solve_kinetics(
      t_span=(t[0], t[-1]),
      P0=args.p0,
      params=params,
      A_func=A_func,
      t_eval=t
    )
    P_model = sol.y[0]
    stats = fit_statistics(P_exp, P_model)

    if stats["RMSE"] < best_rmse:
      best_rmse = stats["RMSE"]
      best_result = {"keff": keff_fit, "n": n_candidate, "kelong": kelong_fit}
      best_P_model = P_model
      best_stats = stats

  # Output
  print(Best fit parameters:)
  print(f" n = {best_result['n']}")
  print(f" keff = {best_result['keff']}")
  print(f" kelong = {best_result['kelong']}")
  print(f" RMSE = {best_stats['RMSE']}", MAE = {best_stats['MAE']}")
  
  header_lines = [
    "time,P_exp,P_model",
    f"n={best_result['n']}",
    f"keff={best_result['keff']}",
    f"kelong={best_result['kelong']}",
    f"RMSE={best_stats['RMSE']}",
    f"MAE={best_stats['MAE']}"
  ]
  header = "\n".join(header_lines)
  
  np.savetxt(
    "model_output.csv",
    np.column_stack((t, P_exp, P_model)),
    delimiter=",",
    header=header,
    comments="# "
  )

if __name__ == "__main__":
  main()
