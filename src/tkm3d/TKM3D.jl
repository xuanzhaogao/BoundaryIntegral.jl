# Truncated-kernel method (TKM) for the free-space 3D Laplace potential of continuous
# volume sources, plus the FINUFFT spread/interp helpers used by `PrecomputedVolumeField`.
# Vendored from TKM3D.jl (MIT, same authors); only the pieces this package uses.
module TKM3D

using FINUFFT
using LinearAlgebra

export ltkm3dc, estimate_kcut3dc
export TKMVals, KCut3DCResult

include("common.jl")
include("spreadonly.jl")
include("continuous.jl")

end
