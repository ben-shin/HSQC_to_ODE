import argparse
import numpy as np
from scipy.optimize import least_squares
from load import load_csv
from i_to_conc import i_to_conc
from ode import dP_dt
from ode import solve_kinetics
from stats import fit_statistics
from fit import residuals

def fit_independent_peaks(t, I_matrix, P0, peak_names, nmin, nmax, kelong_guess, A_func):
  summary = []
  P_model_matrix = []

  for i, peak in enumerate(peak_names):
    I_ratio = I_matrix[:, i]
    P_exp = i_to_conc(I_ratio, P0)

    best_rmse = np.inf
    best_fit = None

    for n_candidate in range(nmin, nmax + 1):
      theta0 = [1e-3, kelong_guess]

      def residuals_fixed(theta, t=t, P_exp=P_exp, P0=P0, n=n_candidate):
        keff, kelong = theta
        return residuals([keff, n, kelong], t, P_exp, P0, A_func)

      result = least_squares(residuals_fixed, theta0, bounds=([0,0],[np.inf,np.inf]))
      keff_fit, kelong_fit = result.x
      params_fit = {"keff": keff_fit, "n": n_candidate, "kelong": kelong_fit}
      sol = solve_kinetics((t[0], t[-1]), P0, params_fit, A_func, t_eval=t)
      P_model = sol.y[0]
      stats = fit_statistics(P_exp, P_model)

      if stats["RMSE"] < best_rmse:
        best_rmse = stats["RMSE"]
        best_fit = {"peak": peak, "n": n_candidate, "keff": keff_fit, "kelong": kelong_fit, "P_model": P_model, "stats": stats}

    summary.append(best_fit)
    P_model_matrix.append(best_fit["P_model"])

  return summary, np.column_stack(P_model_matrix)

def fit_global_peaks(t, I_matrix, P0, peak_names, nmin, nmax, kelong_guess, A_func):
  P_exp_all = np.array([i_to_conc(I_matrix[:, i], P0) for i in range(I_matrix.shape[1])])

  best_rmse_global = np.inf
  best_fit_global = None
  best_P_model_global = None

  # loop over integer n
  for n_candidate in range(nmin, nmax + 1):
    theta0 = [1e-3, kelong_guess]

    def residuals_global(theta):
      keff, kelong = theta
      res = []
      params = {"keff": keff_fit, "n": n_candidate, "kelong": kelong_fit}
      for i in range(P_exp_all.shape[0]):
        sol = solve_kinetics((t[0], t[-1], P0, params_fit, A_func, t_eval=t))
        P_model = sol.y[0]
        P_model_matrix.append(P_model)
        stats = fit_statistics(P_exp_all[i], P_model)
        rmses.append(stats["RMSE"])
      global_rmse = np.mean(rmses)

      if global_rmse < best_rmse_global:
        best_rmse_global = global_rmse
        best_fit_global = {"n": n_candidate, "keff": keff_fit, "kelong": kelong_fit, "stats": {"RMSE": global_rmse}}
        best_P_model_global = np.column_stack(P_model_matrix)

  return best_fit_global, best_P_model_global

def main():
  parser = argparse.ArgumentParser(description="Multi peak kinetic fitting (independent and global)")
  parser.add_argument("--data", type=str, required=True, help="Path to csv")
  parser.add_argument("--p0", type=float, required=True, help="Initial [P]")
  parser.add_argument("--nmin", type=int, default=1, help="Minimum n integer")
  parser.add_argument("--nmax", type=int, default=3, help="Maximum n integer")
  parser.add_argument("--kelong", type=float, default=0.05, help="Initial guess for elongation rate")
  args = parser.parse_args()

  # load csv
  raw = np.genfromtxt(args.data, delimiter=",", names=True)
  t = raw['t']
  peak_names = list(raw.dtype.names[1:])
  I_matrix = np.column_stack([raw[name] for name in peak_names])

  def A_func(t): return 1.0 # placeholder

  # fit independently
  summary_ind, P_model_ind = fit_independent_peaks(t, I_matrix, args.p0, peak_names, args.nmin, args.nmax, args.kelong, A_func)

  # save independent outputs
  header_lines_ind = ["t"] + [f"{peak}_exp,{peak}_model" for peak in peak_names]
  data_to_save_ind = np.column_stack([t] + [val for pair in zip(I_matrix.T, P_model_ind.T) for val in pair])
  np.savetxt("model_output_independent.csv", data_to_save_ind, delimiter=",", header="\n".join(header_lines_ind), comments="")

  with open("fit_summary_independent.csv", "w") as f:
    f.write("peak,n,keff,kelong,RMSE,MAE\n")
    for res in summary_ind:
      f.write(f"{res['peak']},{res['n']},{res['keff']},{res['kelong']},{res['stats']['RMSE']},{res['stats']['MAE']}\n")

  print("Independent fits")
      for res in summary_ind:
      f.write(f"{res['peak']},{res['n']}, keff={res['keff']:.5f}, kelong={res['kelong']:.5f}, RMSE={res['stats']['RMSE']:.2e}")

  # fit globally
  best_fit_global, P_model_global = fit_global_peaks(t, I_matrix, args.p0, peak_names, args.nmin, args.nmax, args.kelong, A_func)

  # save global outputs
  header_lines_global = ["t"] + [f"{peak}_exp,{peak}_model" for peak in peak_names]
  data_to_save_global = np.column_stack([t] + [val for pair in zip(I_matrix.T, P_model_global.T) for val in pair])
  np.savetxt("model_output_global.csv", data_to_save_global, delimiter=",", header="\n".join(header_lines_global), comments="")

  with open("fit_summary_global.csv", "w") as f:
    f.write("n,keff,kelong,RMSE\n")
    f.write(f"{best_fit_global['kelong']:.5f}, RMSE={best_fit_global['stats']['RMSE']:.2e}")

if __name__ == "__main__":
  main()
