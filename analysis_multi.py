import argparse
import numpy as np
import warnings
from scipy.optimize import least_squares

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from load import load_csv
    from i_to_conc import i_to_conc
    from ode import dP_dt, solve_kinetics
    from stats import fit_statistics
    from fit import residuals
except ImportError as e:
    print(f"Module Import Error: {e}. Ensure all helper .py files are in the folder.")

def safe_residuals(theta, t, P_exp, P0, n, A_func):
    """Wraps the residual function to prevent non-finite values from crashing the solver."""
    try:
        res = residuals([theta[0], n, theta[1]], t, P_exp, P0, A_func)
        # If the ODE solver fails or produces NaNs, return a very large penalty
        if not np.all(np.isfinite(res)):
            return np.ones(len(t)) * 1e10
        return res
    except:
        return np.ones(len(t)) * 1e10

def fit_independent_peaks(t, I_matrix, P0, peak_names, nmin, nmax, kelong_guess, A_func):
    summary = []
    P_model_matrix = []

    for i, peak in enumerate(peak_names):
        I_ratio = I_matrix[:, i]
        P_exp = i_to_conc(I_ratio, P0)
        
        # Skip if data is bad
        if not np.all(np.isfinite(P_exp)):
            continue

        best_rmse = np.inf
        best_fit = None

        for n_candidate in range(nmin, nmax + 1):
            # Smaller keff guess (1e-8) prevents the ODE from blowing up at t=0
            theta0 = [1e-8, kelong_guess] 

            try:
                result = least_squares(
                    safe_residuals, 
                    theta0, 
                    args=(t, P_exp, P0, n_candidate, A_func),
                    bounds=([0, 0], [np.inf, np.inf]),
                    ftol=1e-4 # Slight tolerance increase for stability
                )
                
                keff_fit, kelong_fit = result.x
                params_fit = {"keff": keff_fit, "n": n_candidate, "kelong": kelong_fit}
                
                sol = solve_kinetics((t[0], t[-1]), P0, params_fit, A_func, t_eval=t)
                P_model = sol.y[0]
                stats = fit_statistics(P_exp, P_model)

                if stats["RMSE"] < best_rmse:
                    best_rmse = stats["RMSE"]
                    best_fit = {
                        "peak": peak, "n": n_candidate, "keff": keff_fit, 
                        "kelong": kelong_fit, "P_model": P_model, "stats": stats
                    }
            except Exception:
                continue

        if best_fit:
            summary.append(best_fit)
            P_model_matrix.append(best_fit["P_model"])

    return summary, np.column_stack(P_model_matrix)

def fit_global_peaks(t, I_matrix, P0, peak_names, nmin, nmax, kelong_guess, A_func):
    P_exp_all = np.array([i_to_conc(I_matrix[:, i], P0) for i in range(I_matrix.shape[1])])
    
    best_rmse_global = np.inf
    best_fit_global = None
    best_P_model_global = None

    for n_candidate in range(nmin, nmax + 1):
        theta0 = [1e-8, kelong_guess]

        def global_res_wrapper(theta):
            all_res = []
            for i in range(P_exp_all.shape[0]):
                res = safe_residuals(theta, t, P_exp_all[i], P0, n_candidate, A_func)
                all_res.extend(res)
            return np.array(all_res)

        try:
            result = least_squares(global_res_wrapper, theta0, bounds=([0,0],[np.inf,np.inf]))
            keff_fit, kelong_fit = result.x
            
            rmses = []
            current_models = []
            params_final = {"keff": keff_fit, "n": n_candidate, "kelong": kelong_fit}
            
            for i in range(P_exp_all.shape[0]):
                sol = solve_kinetics((t[0], t[-1]), P0, params_final, A_func, t_eval=t)
                current_models.append(sol.y[0])
                stats = fit_statistics(P_exp_all[i], sol.y[0])
                rmses.append(stats["RMSE"])
            
            avg_rmse = np.mean(rmses)
            if avg_rmse < best_rmse_global:
                best_rmse_global = avg_rmse
                best_fit_global = {"n": n_candidate, "keff": keff_fit, "kelong": kelong_fit, "stats": {"RMSE": avg_rmse}}
                best_P_model_global = np.column_stack(current_models)
        except Exception:
            continue

    return best_fit_global, best_P_model_global

def main():
    parser = argparse.ArgumentParser(description="Multi peak kinetic fitting")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--p0", type=float, required=True)
    parser.add_argument("--nmin", type=int, default=1)
    parser.add_argument("--nmax", type=int, default=4)
    parser.add_argument("--kelong", type=float, default=0.00001)
    args = parser.parse_args()

    raw = np.genfromtxt(args.data, delimiter=",", names=True)
    t = raw['time']
    peak_names = list(raw.dtype.names[1:])
    I_matrix = np.column_stack([raw[name] for name in peak_names])

    def A_func(t): return 1.0 

    print(f"Fitting {len(peak_names)} peaks independently...")
    summary_ind, P_model_ind = fit_independent_peaks(t, I_matrix, args.p0, peak_names, args.nmin, args.nmax, args.kelong, A_func)
    
    with open("fit_summary_independent.csv", "w") as f:
        f.write("peak,n,keff,kelong,RMSE,MAE\n")
        for res in summary_ind:
            f.write(f"{res['peak']},{res['n']},{res['keff']:.5e},{res['kelong']:.5e},{res['stats']['RMSE']:.2e},{res['stats']['MAE']:.2e}\n")

    print("Running Global Fit...")
    best_fit_global, P_model_global = fit_global_peaks(t, I_matrix, args.p0, peak_names, args.nmin, args.nmax, args.kelong, A_func)

    if best_fit_global:
        with open("fit_summary_global.csv", "w") as f:
            f.write("n,keff,kelong,RMSE\n")
            f.write(f"{best_fit_global['n']},{best_fit_global['keff']:.5e},{best_fit_global['kelong']:.5e},{best_fit_global['stats']['RMSE']:.2e}\n")
        print(f"Global Fit Results: n={best_fit_global['n']}, kelong={best_fit_global['kelong']:.2e}")
    else:
        print("Global fit failed to converge.")

if __name__ == "__main__":
    main()
