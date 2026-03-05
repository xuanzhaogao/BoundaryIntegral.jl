module BoundaryIntegral

using LinearAlgebra, Statistics
using SparseArrays, StaticArrays
using FastGaussQuadrature, LegendrePolynomials, SpecialFunctions
using Krylov, LinearMaps, Roots, HCubature
using NearestNeighbors
using ForwardDiff

using FMM2D, FMM3D
using FBCPoisson

#core types
export FlatPanel
export DielectricInterface
export rhs_approx
export interface_approx
export interface_uniform_samples

export PointSource, VolumeSource

# kernel functions
export laplace2d_pot, laplace2d_grad
export laplace2d_S, laplace2d_D, laplace2d_DT, laplace2d_pottrg
export laplace2d_S_fmm2d, laplace2d_DT_fmm2d, laplace2d_D_fmm2d, laplace2d_pottrg_fmm2d

export laplace3d_pot, laplace3d_grad
export laplace3d_S, laplace3d_D, laplace3d_DT, laplace3d_pottrg
export laplace3d_S_fmm3d, laplace3d_DT_fmm3d, laplace3d_D_fmm3d, laplace3d_pottrg_fmm3d
export laplace3d_DT_fmm3d_corrected
export laplace3d_DT_fmm3d_corrected_hcubature
export laplace3d_D_fmm3d_corrected
export laplace3d_pottrg_fmm3d_corrected_hcubature
export laplace3d_pottrg_near

# shapes
export single_dielectric_box2d, multi_dielectric_box2d
export single_dielectric_box3d
export single_dielectric_box3d_rhs_adaptive
export single_dielectric_box3d_rhs_adaptive_varquad

# solvers
export Lhs_dielectric_box2d, Lhs_dielectric_box2d_fmm2d, Rhs_dielectric_box2d
export Lhs_dielectric_box3d, Lhs_dielectric_box3d_fmm3d, Rhs_dielectric_box3d, Rhs_dielectric_box3d_fmm3d, Rhs_dielectric_box3d_hybrid

# linear algebra
export solve_lu, solve_gmres

# visualization
export viz_2d, viz_3d
export viz_3d_surface
export viz_3d_interface_solution
export viz_3d_zslice

# core types
include("core/panels.jl")
include("core/sources.jl")

# kernel functions
include("kernel/laplace2d.jl")
include("kernel/laplace3d.jl")
include("kernel/laplace3d_near.jl")

# # geometries
include("shape/box2d.jl")
include("shape/box3d.jl")

# # utilities
include("utils/linear_algebra.jl")
include("utils/corner_singularity.jl")
include("utils/barycentric.jl")
include("utils/bernstein.jl")
include("utils/quad_order.jl")
include("utils/best_grid.jl")
include("utils/gaussians.jl")
include("utils/xsf_reader.jl")

# # solvers
include("solver/dielectric_box2d.jl")
include("solver/dielectric_box3d.jl")

# # visualization
include("visualization/viz_2d.jl")
include("visualization/viz_3d.jl")

end
