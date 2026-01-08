import argparse
import numpy as np
import warnings
import sys
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")

try:
    from i_to_conc import i_to_conc
    from ode import solve_kinetics
    from stats import fit_statistics
    from fit import residuals
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def safe_residuals(theta, t, P_exp, n, A_func, model_type):
    keff, kelong, krise = theta
    # We use P0=1.0 for normalized fitting
    params = {'keff': keff, 'n': n, 'kelong': kelong, 'krise': krise}
    
    if model_type == 'standard':
        params['krise'] = 1e5
        params['f_visible'] = 1.0
    else:
        params['f_visible'] = P_exp[0]

    try:
        sol = solve_kinetics((t[0], t[-1]), 1.0, params, A_func, t_eval=t)
        if not sol.success:
            return np.ones(len(t)) * 10 # High penalty
        
        P_model = sol.y[1]
        res = P_exp - P_model
        return res if np.all(np.isfinite(res)) else np.ones(len(t)) * 10
    except:
        return np.ones(len(t)) * 10

def fit_independent_peaks(t, I_matrix, P0, peak_names, nmin, nmax, kelong_guess, A_func):
    summary = []
    P_model_matrix = []

    for i, peak in enumerate(peak_names):
        I_raw = I_matrix[:, i]
        max_val = np.max(I_raw)
        if max_val <= 0: continue
        
        # Working in Normalized space (0 to 1)
        P_exp = I_raw / max_val 
        model_type = 'rise_fall' if max_val > I_raw[0] * 1.05 else 'standard'

        best_rmse = np.inf
        best_fit = None

        for n_cand in range(nmin, nmax + 1):
            # Broader search: try multiple guesses if the first one fails
            for guess_scale in [1e-10, 1e-8, 1e-6]:
                theta0 = [guess_scale, kelong_guess, 1e-4]
                try:
                    res_lsq = least_squares(
                        safe_residuals, theta0, 
                        args=(t, P_exp, n_cand, A_func, model_type),
                        bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
                        ftol=1e-2 # Looser tolerance to encourage convergence
                    )
                    
                    keff_f, kelong_f, krise_f = res_lsq.x
                    p_final = {'keff': keff_f, 'n': n_cand, 'kelong': kelong_f, 
                               'krise': krise_f if model_type == 'rise_fall' else 1e5,
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
                        break # Found a working fit
                except Exception as e:
                    if i == 0: print(f"Diagnostic for {peak}: Guess {guess_scale} failed: {e}")
                    continue

        if best_fit:
            summary.append(best_fit)
            P_model_matrix.append(best_fit["P_model"])

    if not summary:
        print("\nCRITICAL: All residues failed.")
        print(f"Time Range: {t[0]} to {t[-1]}")
        print(f"Data Sample (First 5 points of {peak_names[0]}): {I_matrix[:5, 0] / np.max(I_matrix[:, 0])}")
        sys.exit(1)

    return summary, np.column_stack(P_model_matrix)

def main():
    parser = argparse.ArgumentParser()
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

    print(f"Success. Fitted {len(summary_ind)} peaks.")

if __name__ == "__main__":
    main()
