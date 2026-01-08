import argparse
import numpy as np
import pandas as pd
import warnings
import sys
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")

try:
    from ode import solve_kinetics
    from stats import fit_statistics
except ImportError as e:
    print(f"Import Error: {e}. Ensure ode.py and stats.py are in this folder.")
    sys.exit(1)

def safe_residuals(theta, t, P_exp, n, A_func, model_type):
    keff, kelong, krise = theta
    params = {'keff': keff, 'n': n, 'kelong': kelong, 'krise': krise}
    if model_type == 'standard':
        params['krise'] = 1e8
        params['f_visible'] = 1.0
    else:
        params['f_visible'] = P_exp[0]

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
    total = len(peak_names)
    for i, peak in enumerate(peak_names):
        I_raw = I_matrix[:, i]
        
        # Skip if array is empty or all NaNs
        if I_raw.size == 0 or np.all(np.isnan(I_raw)):
            continue
            
        max_val = np.nanmax(I_raw)
        if max_val <= 0: continue
        
        P_exp = I_raw / max_val 
        # Use Rise/Fall if peak is 5% higher than starting point
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
                except: continue
        if best_fit: 
            summary.append(best_fit)
        
        if (i+1) % 20 == 0:
            print(f"  Progress: {i+1}/{total} peaks...")
            
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--p0", type=float, required=True)
    parser.add_argument("--nmin", type=int, default=1)
    parser.add_argument("--nmax", type=int, default=4)
    parser.add_argument("--kelong", type=float, default=1e-5)
    args = parser.parse_args()

    try:
        # Load the CSV. header=0 makes the first row (the times) the columns.
        # index_col=0 makes the first column (the residue names) the rows.
        df = pd.read_csv(args.data, index_col=0)

        # Transpose so Rows = Time, Columns = Residue names
        df = df.T
        
        # Clean the index (convert timestamps from strings/headers to floats)
        df.index = pd.to_numeric(df.index, errors='coerce')
        
        # Drop the row that likely says 'time' in the index if it exists
        df = df[df.index.notnull()]
        df = df.sort_index()
        
        # Clean the intensity data
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.interpolate(method='linear', axis=0).fillna(method='bfill').fillna(method='ffill')

        t = df.index.values
        peak_names = df.columns.tolist()
        I_matrix = df.values.astype(float)
        
        print(f"Successfully Loaded:")
        print(f"- {len(t)} Timepoints (from {t[0]} to {t[-1]} seconds)")
        print(f"- {len(peak_names)} Residues")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    if len(t) == 0:
        print("Error: No timepoints found. Check the format of your CSV headers.")
        sys.exit(1)

    def A_func(t): return 1.0 
    print(f"Starting fitting...")
    summary = fit_independent_peaks(t, I_matrix, peak_names, args.nmin, args.nmax, args.kelong, A_func)
    
    if not summary:
        print("All fits failed. Check your initial guesses.")
        sys.exit(1)

    with open("fit_summary_independent.csv", "w") as f:
        f.write("peak,type,n,keff,kelong,krise,RMSE\n")
        for res in summary:
            f.write(f"{res['peak']},{res['type']},{res['n']},{res['keff']:.3e},{res['kelong']:.3e},{res['krise']:.3e},{res['stats']['RMSE']:.2e}\n")
    print(f"Success! Results saved to fit_summary_independent.csv")

if __name__ == "__main__":
    main()
