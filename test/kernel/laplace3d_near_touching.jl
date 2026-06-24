using BoundaryIntegral
import BoundaryIntegral as BI
using FastGaussQuadrature, HCubature, LinearAlgebra, StaticArrays
using Test

# Cross-face touching pair. Compute (DT σ)(t) at a target t on the
# touching panel via:
#   (a) HCubature over both panels (reference),
#   (b) sparse corrections with correct_edges = false,
#   (c) sparse corrections with correct_edges = true.
# Expect (c) much closer to (a) than (b).
@testset "cross-face touching DT regression" begin
    n_quad = 8
    ns, ws = gausslegendre(n_quad); ns = Float64.(ns); ws = Float64.(ws)

    # panel_a in z = 0 plane, x in [-0.5, 0.5], y in [-0.5, 0.5].
    a1 = (-0.5, -0.5, 0.0); b1 = ( 0.5, -0.5, 0.0)
    c1 = ( 0.5,  0.5, 0.0); d1 = (-0.5,  0.5, 0.0)
    n1 = (0.0, 0.0, 1.0)
    panel_a = BI.rect_panel3d_discretize(a1, b1, c1, d1, ns, ws, n1)

    # panel_b in x = 0.5 plane, sharing the edge with panel_a.
    a2 = ( 0.5, -0.5, 0.0); b2 = ( 0.5,  0.5, 0.0)
    c2 = ( 0.5,  0.5, 1.0); d2 = ( 0.5, -0.5, 1.0)
    n2 = (1.0, 0.0, 0.0)
    panel_b = BI.rect_panel3d_discretize(a2, b2, c2, d2, ns, ws, n2)

    interface = BI.DielectricInterface([panel_a, panel_b], [1.0, 1.0], [1.0, 1.0])

    # Smooth density.
    σ(p) = exp(-2.0 * (p[1]^2 + p[2]^2 + p[3]^2))
    np = BI.num_points(interface)
    σvec = Vector{Float64}(undef, np)
    k = 0
    for panel in interface.panels
        for p in panel.points
            k += 1; σvec[k] = σ(p)
        end
    end

    # Reference (DT σ)(t) at a target point on panel_b near the shared edge.
    panel_b_idx = length(panel_a.points)            # offset for panel_b targets
    t_local = 1                                     # first GL node on panel_b
    t_global = panel_b_idx + t_local
    target = panel_b.points[t_local]
    target_normal = panel_b.normal

    # HCubature reference: integrate over both panels using the same parametrization
    # used elsewhere in this codebase.
    function hcub_DT(panel, target, target_normal)
        a, b, c, d = panel.corners
        cc = (a .+ b .+ c .+ d) ./ 4
        bma = b .- a; dma = d .- a
        Lx = norm(bma); Ly = norm(dma); s = Lx * Ly / 4
        integrand(uv) = begin
            u, v = uv[1], uv[2]
            y = cc .+ bma .* (u / 2) .+ dma .* (v / 2)
            σ(y) * BI.laplace3d_grad(y, target, target_normal) * s
        end
        val, _ = hcubature(integrand,
                           SVector{2,Float64}(-1.0, -1.0),
                           SVector{2,Float64}( 1.0,  1.0); atol = 1e-10)
        return val
    end
    ref_val = hcub_DT(panel_a, target, target_normal) + hcub_DT(panel_b, target, target_normal)

    # Apply the FMM+correction operator to σvec, twice.
    op_off = BI.laplace3d_DT_fmm3d_corrected(
        interface, 1e-10, 1e-8, 24;
        correct_edges = false)
    op_on  = BI.laplace3d_DT_fmm3d_corrected(
        interface, 1e-10, 1e-8, 24;
        correct_edges = true,  adaptive_atol = 1e-8)

    y_off = op_off * σvec
    y_on  = op_on  * σvec

    err_off = abs(y_off[t_global] - ref_val)
    err_on  = abs(y_on[t_global]  - ref_val)

    @info "cross-face touching DT: ref_val=$ref_val, err_off=$err_off, err_on=$err_on"
    @info "cross-face DT errors" err_off err_on

    @test err_on < err_off
    @test err_on < 1e-5    # loose acceptance: adaptive should reach reasonable accuracy
end
