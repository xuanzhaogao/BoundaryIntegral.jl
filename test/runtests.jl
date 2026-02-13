using BoundaryIntegral
using LinearAlgebra
using Test

@testset "BoundaryIntegral.jl" begin


    # utilities
    include("utils/linear_algebra.jl")
    include("utils/barycentric.jl")
    include("utils/quad_order.jl")
    include("utils/best_grid.jl")

    # core
    include("core/panels.jl")
    include("core/sources.jl")

    # kernel functions
    include("kernel/laplace2d.jl")
    include("kernel/laplace3d.jl")
    include("kernel/laplace3d_near.jl")

    # shape functions
    include("shape/box3d.jl")

    # solver functions
    include("solver/dielectric_box2d.jl")
    include("solver/dielectric_box3d.jl")
end
