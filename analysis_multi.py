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
        
        if I_raw.size == 0 or np.all(np.isnan(I_raw)):
            continue
            
        max_val = np.nanmax(I_raw)
        if max_val <= 0 or np.isnan(max_val): continue
        
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
                except: continue
        if best_fit: 
            summary.append(best_fit)
        if (i+1) % 20 == 0:
            print(f"  Progress: {i+1}/{total} peaks fitted...")
            
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
        # Load the CSV. header=0 (first row is time), index_col=0 (first column is residue)
        df = pd.read_csv(args.data, index_col=0)

        # Transpose so index = Time, columns = Residue
        df = df.T
        
        # Clean the index (convert timestamps from headers to floats)
        # We drop any index values that cannot be converted to numbers (like 'time')
        df.index = pd.to_numeric(df.index, errors='coerce')
        df = df[df.index.notnull()]
        
        # Clean the data (ensure all values are floats)
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # Interpolate missing points
        df = df.interpolate(method='linear', axis=0).fillna(method='bfill').fillna(method='ffill')

        # FINAL CHECK: Ensure we actually have data
        if df.empty:
            raise ValueError("The resulting dataframe is empty after numeric conversion.")

        t = df.index.values.astype(float)
        peak_names = df.columns.tolist()
        I_matrix = df.values.astype(float)
        
        # Check for monotonicity (Time must only increase)
        if not np.all(np.diff(t) > 0):
            print("Warning: Time points were not sorted. Sorting now...")
            idx = np.argsort(t)
            t = t[idx]
            I_matrix = I_matrix[idx, :]

        print(f"Successfully Loaded:")
        print(f"- {len(t)} Timepoints (Range: {t[0]} to {t[-1]} seconds)")
        print(f"- {len(peak_names)} Residues")
    except Exception as e:
        print(f"Fatal Error: {e}")
        sys.exit(1)

    def A_func(t): return 1.0 
    print(f"Starting fits...")
    summary = fit_independent_peaks(t, I_matrix, peak_names, args.nmin, args.nmax, args.kelong, A_func)
    
    if not summary:
        print("All fits failed. Please check your data scaling.")
        sys.exit(1)

    # Save to CSV
    res_df = pd.DataFrame([
        {
            "peak": r["peak"], "type": r["type"], "n": r["n"], 
            "keff": r["keff"], "kelong": r["kelong"], "krise": r["krise"], 
            "RMSE": r["stats"]["RMSE"]
        } for r in summary
    ])
    res_df.to_csv("fit_summary_independent.csv", index=False)
    
    print(f"Success! Results saved to fit_summary_independent.csv")

if __name__ == "__main__":
    main()
