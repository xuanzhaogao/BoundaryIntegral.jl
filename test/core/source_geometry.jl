using Test
using LinearAlgebra
using BoundaryIntegral
const BI = BoundaryIntegral

@testset "VolumeSource lattice_basis" begin
    # --- identity-basis grid ctor: axes hold ABSOLUTE coords with step h ---
    n = 8
    h = 2.0 / n
    xs = collect(-1.0 + h/2 .+ h .* (0:n-1))
    w  = fill(h^3, n, n, n)
    d  = fill(1.0, n, n, n)
    vs = VolumeSource((xs, xs, xs), w, d)
    @test vs.lattice_basis !== nothing
    # A_rho = h * I
    for j in 1:3, i in 1:3
        @test isapprox(vs.lattice_basis[j][i], i == j ? h : 0.0; atol = 1e-14)
    end

    # --- skew ctor: axes hold FRACTIONAL coords, basis holds full cell vectors ---
    nx = 4
    frac = collect((i - 1) / nx for i in 1:nx)
    At = (2.0, 0.0, 0.0); Bt = (1.0, 2.0, 0.0); Ct = (0.0, 0.0, 3.0)
    wg = fill(1.0, nx, nx, nx); dg = fill(1.0, nx, nx, nx)
    vsk = VolumeSource((frac, frac, frac), wg, dg, (0.0, 0.0, 0.0), (At, Bt, Ct))
    # primitive step vectors are the cell vectors divided by nx
    @test isapprox(collect(vsk.lattice_basis[1]), collect(At) ./ nx; atol = 1e-14)
    @test isapprox(collect(vsk.lattice_basis[2]), collect(Bt) ./ nx; atol = 1e-14)
    @test isapprox(collect(vsk.lattice_basis[3]), collect(Ct) ./ nx; atol = 1e-14)

    # --- flat ctor from bare points: no lattice ---
    pts = rand(3, 10)
    vsf = VolumeSource(pts, fill(1.0, 10), fill(1.0, 10))
    @test vsf.lattice_basis === nothing

    # --- flat ctor accepts an explicit basis ---
    lb_explicit = ((h, 0.0, 0.0), (0.0, h, 0.0), (0.0, 0.0, h))
    vsb = VolumeSource(pts, fill(1.0, 10), fill(1.0, 10);
                       lattice_basis = lb_explicit)
    @test vsb.lattice_basis == lb_explicit

    # --- with_density preserves the basis ---
    vs2 = BI.with_density(vs, fill(2.0, length(vs.density)))
    @test vs2.lattice_basis == vs.lattice_basis
    @test all(vs2.density .== 2.0)
    @test vs2.positions == vs.positions

    # --- non-uniform axes are not a lattice ---
    bad = [0.0, 0.1, 0.5, 1.0]
    vsn = VolumeSource((bad, bad, bad), fill(1.0, 4, 4, 4), fill(1.0, 4, 4, 4))
    @test vsn.lattice_basis === nothing
end

@testset "screened_volume_source preserves lattice_basis" begin
    gsrc = BI.GaussianVolumeSource((0.0, 0.0, 0.0), 0.3, 8, 1e-6)
    @test gsrc.lattice_basis !== nothing

    # --- multibox overload: (boxes, epses, eps_out, vs, mode) ---
    boxes = [(center = (0.0, 0.0, 0.0), Lx = 1.0, Ly = 1.0, Lz = 1.0)]
    sc = BI.screened_volume_source(boxes, [2.0], 1.0, gsrc, BI.SharpScreening())
    @test sc.lattice_basis == gsrc.lattice_basis

    # --- single-box overload: (Lx, Ly, Lz, vs, eps_in, eps_out, mode) ---
    sc_box = BI.screened_volume_source(1.0, 1.0, 1.0, gsrc, 2.0, 1.0, BI.SharpScreening())
    @test sc_box.lattice_basis == gsrc.lattice_basis

    # --- interface overload: (interface, vs, mode) ---
    # Cheapest available fixture: a uniform-eps box interface built the same way
    # test/core/panels.jl does, via single_dielectric_box3d (single call, no
    # manual panel/quadrature setup).
    iface = BI.single_dielectric_box3d(1.0, 1.0, 1.0, 3, 0.3, 2.0, 1.0)
    sc_iface = BI.screened_volume_source(iface, gsrc, BI.SharpScreening())
    @test sc_iface.lattice_basis == gsrc.lattice_basis
end
