# HSQC_to_ODE
Takes I/I0 of amyloids over time and solves for aggregation kinetic constants.  
This was purposely written to be modular.  

## Installation
```
git clone https://github.com/ben-shin/HSQC_to_ODE.git
```

## Usage
```
python analysis.py --data 'path/to/csv' --p0 'Initial protein concentration' --nmin 'integer value minimum reaction order' --nmax 'integer value maximum reaction order' --kelong 'elongation rate'
python analysis_multi.py --data 'path/to/csv' --p0 'Initial protein concentration' --nmin 'integer value minimum reaction order' --nmax 'integer value maximum reaction order' --kelong 'elongation rate'
```

## Data Format
Input data should be a CSV file with headers. It doesn't matter what they are but they should be ordered as such:  
time (seconds), residue 1, residue 2...
