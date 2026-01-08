import argparse
import numpy as np
import warnings
import sys
import pandas as pd # Adding pandas for much more robust CSV reading
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")

try:
    from ode import solve_kinetics
    from stats import fit_statistics
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def safe_residuals(theta, t, P_exp, n, A_func, model_type):
    keff, kelong, krise = theta
    params = {'keff': keff, 'n': n, 'kelong': kelong, 'krise': krise}
    params['f_visible'] = 1.0 if model_type == 'standard' else P_exp[0]
    if model_type == 'standard': params['krise'] = 1e8

    try:
        sol = solve_kinetics((t[0], t[-1]), 1.0, params, A_func, t_eval=t)
        if not sol.success: return np.ones(len(t)) * 10
        P_model = sol.y[1]
        res = P_exp - P_model
        return res if np.all(np.isfinite(res)) else np.ones(len(t)) * 10
    except:
        return np.ones(len(t)) * 10

def fit_independent_peaks(t, I_matrix, peak_names, nmin, nmax, kelong_guess, A_func):
    summary = []
    
    for i, peak in enumerate(peak_names):
        I_raw = I_matrix[:, i]
        
        # Final safety check for NaNs in this specific column
        if np.any(np.isnan(I_raw)):
            continue
            
        max_val = np.max(I_raw)
        P_exp = I_raw / max_val 
        model_type = 'rise_fall' if max_val > I_raw[0] * 1.05 else 'standard'

        best_rmse = np.inf
        best_fit = None

        for n_cand in range(nmin, nmax + 1):
            for guess_scale in [1e-10, 1e-8, 1e-6]:
                theta0 = [guess_scale, kelong_guess, 1e-4]
                try:
                    res_lsq = least_squares(
                        safe_residuals, theta0, 
                        args=(t, P_exp, n_cand, A_func, model_type),
                        bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
                        ftol=1e-2
                    )
                    kef, kel, kri = res_lsq.x
                    p_f = {'keff': kef, 'n': n_cand, 'kelong': kel, 
                           'krise': kri if model_type == 'rise_fall' else 1e8,
                           'f_visible': P_exp[0] if model_type == 'rise_fall' else 1.0}
                    
                    sol = solve_kinetics((t[0], t[-1]), 1.0, p_f, A_func, t_eval=t)
                    stats = fit_statistics(P_exp, sol.y[1])

                    if stats["RMSE"] < best_rmse:
                        best_rmse = stats["RMSE"]
                        best_fit = {"peak": peak, "n": n_cand, "keff": kef, "kelong": kel, 
                                    "krise": p_f['krise'], "type": model_type, "stats": stats}
                        break
                except:
                    continue

        if best_fit:
            summary.append(best_fit)

    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--p0", type=float, required=True)
    parser.add_argument("--nmin", type=int, default=1)
    parser.add_argument("--nmax", type=int, default=4)
    parser.add_argument("--kelong", type=float, default=1e-5)
    args = parser.parse_args()

    # --- Robust Data Loading with Pandas ---
    try:
        df = pd.read_csv(args.data)
        # Convert any "None" strings to actual NaNs and drop those columns
        df = df.replace('None', np.nan).dropna(axis=1, how='any')
        
        # Ensure time is sorted (important for ODE solver)
        df = df.sort_values(by='time')
        
        t = df['time'].values
        peak_names = [col for col in df.columns if col != 'time']
        I_matrix = df[peak_names].values
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    if len(peak_names) == 0:
        print("No valid residues found after cleaning NaNs.")
        sys.exit(1)

    def A_func(t): return 1.0 

    print(f"Processing {len(peak_names)} valid peaks...")
    summary = fit_independent_peaks(t, I_matrix, peak_names, args.nmin, args.nmax, args.kelong, A_func)
    
    if not summary:
        print("All fits failed. Check if time units match kelong guess.")
        sys.exit(1)

    with open("fit_summary_independent.csv", "w") as f:
        f.write("peak,type,n,keff,kelong,krise,RMSE\n")
        for res in summary:
            f.write(f"{res['peak']},{res['type']},{res['n']},{res['keff']:.3e},{res['kelong']:.3e},{res['krise']:.3e},{res['stats']['RMSE']:.2e}\n")

    print(f"Success. Results saved for {len(summary)} residues.")

if __name__ == "__main__":
    main()
