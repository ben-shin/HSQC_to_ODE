import argparse
import numpy as np
import warnings
import sys
from scipy.optimize import least_squares

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from i_to_conc import i_to_conc
    from ode import solve_kinetics
    from stats import fit_statistics
    from fit import residuals
except ImportError as e:
    print(f"Import Error: {e}. Check helper files.")
    sys.exit(1)

def safe_residuals(theta, t, P_exp, P0, n, A_func, model_type):
    keff, kelong, krise = theta
    # Note: We use P0=1.0 here because P_exp is normalized 0->1
    params = {'keff': keff, 'n': n, 'kelong': kelong, 'krise': krise}
    
    if model_type == 'standard':
        params['krise'] = 1e8 # Instant rise
        params['f_visible'] = 1.0
    else:
        params['f_visible'] = P_exp[0] # Initial normalized visible fraction

    try:
        sol = solve_kinetics((t[0], t[-1]), 1.0, params, A_func, t_eval=t)
        if not sol.success:
            return np.ones(len(t)) * 1e10
        
        P_model = sol.y[1]
        res = P_exp - P_model
        return res if np.all(np.isfinite(res)) else np.ones(len(t)) * 1e10
    except Exception as e:
        return np.ones(len(t)) * 1e10

def fit_independent_peaks(t, I_matrix, P0, peak_names, nmin, nmax, kelong_guess, A_func):
    summary = []
    P_model_matrix = []

    for i, peak in enumerate(peak_names):
        I_raw = I_matrix[:, i]
        max_val = np.max(I_raw)
        if max_val == 0: continue
        
        model_type = 'rise_fall' if max_val > I_raw[0] * 1.05 else 'standard'
        
        # Working in Normalized space (0 to 1) for stability
        P_exp = I_raw / max_val 

        best_rmse = np.inf
        best_fit = None

        for n_cand in range(nmin, nmax + 1):
            # Try a very small keff to start
            theta0 = [1e-10, kelong_guess, 1e-4]
            bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])

            try:
                res_lsq = least_squares(
                    safe_residuals, theta0, 
                    args=(t, P_exp, 1.0, n_cand, A_func, model_type),
                    bounds=bounds, ftol=1e-3
                )
                
                keff_f, kelong_f, krise_f = res_lsq.x
                p_final = {'keff': keff_f, 'n': n_cand, 'kelong': kelong_f, 
                           'krise': krise_f if model_type == 'rise_fall' else 1e8,
                           'f_visible': P_exp[0] if model_type == 'rise_fall' else 1.0}
                
                sol = solve_kinetics((t[0], t[-1]), 1.0, p_final, A_func, t_eval=t)
                P_model = sol.y[1]
                stats = fit_statistics(P_exp, P_model)

                if stats["RMSE"] < best_rmse:
                    best_rmse = stats["RMSE"]
                    best_fit = {
                        "peak": peak, "n": n_cand, "keff": keff_f, "kelong": kelong_f, 
                        "krise": p_final['krise'], "type": model_type, 
                        "P_model": P_model, "stats": stats
                    }
            except Exception as e:
                # Print the error for the first peak to debug
                if i == 0: print(f"Debug: Fit failed for {peak} due to {e}")
                continue

        if best_fit:
            summary.append(best_fit)
            P_model_matrix.append(best_fit["P_model"])

    if not P_model_matrix:
        print("Error: No peaks were successfully fitted. Check your ODE parameters.")
        sys.exit(1)

    return summary, np.column_stack(P_model_matrix)

def main():
    parser = argparse.ArgumentParser(description="FapC Rise and Fall Kinetic Fitting")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--p0", type=float, required=True)
    parser.add_argument("--nmin", type=int, default=1)
    parser.add_argument("--nmax", type=int, default=4)
    parser.add_argument("--kelong", type=float, default=1e-5)
    args = parser.parse_args()

    raw = np.genfromtxt(args.data, delimiter=",", names=True)
    t = raw['time']
    peak_names = list(raw.dtype.names[1:])
    I_matrix = np.column_stack([raw[name] for name in peak_names])

    def A_func(t): return 1.0 

    print(f"Processing {len(peak_names)} peaks...")
    summary_ind, P_model_ind = fit_independent_peaks(t, I_matrix, args.p0, peak_names, args.nmin, args.nmax, args.kelong, A_func)
    
    with open("fit_summary_independent.csv", "w") as f:
        f.write("peak,type,n,keff,kelong,krise,RMSE\n")
        for res in summary_ind:
            f.write(f"{res['peak']},{res['type']},{res['n']},{res['keff']:.3e},{res['kelong']:.3e},{res['krise']:.3e},{res['stats']['RMSE']:.2e}\n")

    print(f"Done. Successfully fitted {len(summary_ind)} peaks.")

if __name__ == "__main__":
    main()
