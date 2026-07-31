#!/usr/bin/env bash
# 2x2 sweep for examples/diag_precond_experiment.jl: {edge correction on, off} x
# {plain, preconditioned}, over the ε₂ contrast sweep. Run on a 96-core node, e.g.
#   ssh worker7054 'bash /mnt/home/xgao1/codes/BoundaryIntegral.jl/examples/run_precond_2x2.sh'
set -euo pipefail

REPO=/mnt/home/xgao1/codes/BoundaryIntegral.jl
JULIA=/mnt/home/xgao1/.juliaup/bin/julia
NT=${NT:-96}

cd "$REPO"

export OMP_NUM_THREADS=$NT
export OMP_PROC_BIND=spread
export OPENBLAS_NUM_THREADS=1

export BI_PC_NQUAD=${BI_PC_NQUAD:-6}
export BI_PC_RHS_TOL=${BI_PC_RHS_TOL:-1e-3}
export BI_PC_LEC=${BI_PC_LEC:-1.25}
export BI_PC_MAXDEPTH=${BI_PC_MAXDEPTH:-12}
export BI_PC_EPS2=${BI_PC_EPS2:-6,20,60,200}

echo "host=$(hostname) cwd=$(pwd) threads=$NT"
for CE in 1 0; do
    echo "===== CORRECT_EDGES=$CE"
    BI_PC_CORRECT_EDGES=$CE "$JULIA" --project=. -t "$NT" examples/diag_precond_experiment.jl
done
