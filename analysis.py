import argparse
import numpy as np
from load import load_csv
from i_to_conc import i_to_conc
from ode import dP_dt
from ode import solve_kinetics
from stats import fit_statistics

def main():
  parser = argparse.ArugmentParser(
    description="Kinetic analysis of I/I0 timeseries"
  )

  parser.add_argument(
    "--data",
    type=str,
    required=True
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
  t, I_ratio = load_csv("data.csv")
  
  # convert to conc
  p_exp = i_to_conc(I_ratio, args.p0)
  
  # define A(t)
  def A_func(t):
    return 1.0 # just a placeholder
  
  # solve ODE
  params = {
    "keff": 1e-3,
    "n": 2,
    "kelong": 5e-2
  }
  
  sol = solve_kinetics(
    t_span=(t[0], t[-1]),
    P0=P0,
    params=params,
    A_func=A_func,
    t_eval=t
  )
  
  P_model = sol.y[0]

  stats = fit_statistics(P_exp, P_model)

  # Output
  header_lines = [
    "time,P_exp,P_model",
    f"RMSE={stats['RMSE']}",
    f"MAE={stats['MAE']}"
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
