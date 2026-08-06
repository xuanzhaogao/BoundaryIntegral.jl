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

@testset "near_field_geometry matches Section 3 by hand" begin
    # Cell-centered cubic grid on B = [-1,1]^3, so l = 2 exactly.
    n = 64
    h = 2.0 / n
    xs = collect(-1.0 + h/2 .+ h .* (0:n-1))
    w  = fill(h^3, n, n, n)
    d  = fill(1.0, n, n, n)
    vs = VolumeSource((xs, xs, xs), w, d)

    @test isapprox(BI.lattice_spacing(vs), h; rtol = 1e-14)

    lo, hi, l = BI.source_box(vs)
    for a in 1:3
        @test isapprox(lo[a], -1.0; atol = 1e-14)
        @test isapprox(hi[a],  1.0; atol = 1e-14)
        @test isapprox(l[a],   2.0; atol = 1e-14)
    end

    c_pad = 5.0
    g = BI.near_field_geometry(vs; c_pad = c_pad)
    hn_exp = c_pad * h
    L_exp  = sqrt(3 * (2.0 + hn_exp)^2)
    @test isapprox(g.hn, hn_exp; rtol = 1e-14)
    @test isapprox(g.L,  L_exp;  rtol = 1e-14)
    for a in 1:3
        @test isapprox(g.lo[a], -1.0 - hn_exp; atol = 1e-14)
        @test isapprox(g.hi[a],  1.0 + hn_exp; atol = 1e-14)
        @test isapprox(g.center[a], 0.0; atol = 1e-14)
        # Eq. (3.16) at equality
        @test isapprox(g.dks[a], 2π / (2.0 + hn_exp + L_exp); rtol = 1e-12)
        # prevfloat keeps us strictly inside the aliasing-free set
        @test g.dks[a] <= 2π / (2.0 + hn_exp + L_exp)
    end

    # Eq. (3.9) box test
    tg = [0.0  1.0 + hn_exp/2   1.0 + 2*hn_exp;
          0.0  0.0              0.0;
          0.0  0.0              0.0]
    @test BI.in_near_region(g, tg, 1)
    @test BI.in_near_region(g, tg, 2)
    @test !BI.in_near_region(g, tg, 3)

    # Eq. (3.9) is inclusive: exactly on the hi[1] face is near, one ULP past is not.
    tg_edge = [g.hi[1]              nextfloat(g.hi[1]);
               g.center[2]          g.center[2];
               g.center[3]          g.center[3]]
    @test BI.in_near_region(g, tg_edge, 1)
    @test !BI.in_near_region(g, tg_edge, 2)
end

@testset "lattice_spacing is the 2-norm, not the shortest step" begin
    # Skew lattice: ||A_rho||_2 must exceed the longest column norm.
    nx = 4
    frac = collect((i - 1) / nx for i in 1:nx)
    At = (1.0, 0.0, 0.0); Bt = (0.9, 0.4, 0.0); Ct = (0.0, 0.0, 1.0)
    vsk = VolumeSource((frac, frac, frac), fill(1.0, nx, nx, nx), fill(1.0, nx, nx, nx),
                       (0.0, 0.0, 0.0), (At, Bt, Ct))
    A = hcat(collect.(collect(vsk.lattice_basis))...)
    @test isapprox(BI.lattice_spacing(vsk), opnorm(A, 2); rtol = 1e-12)
    @test BI.lattice_spacing(vsk) > maximum(norm.(collect.(collect(vsk.lattice_basis))))

    # source_box half-extent must be a row-sum over lattice vectors' component
    # along each fixed axis, not a column-sum (which would be invisible on a
    # diagonal/cubic basis). Hand-derived from A_rho columns 0.25*At = (0.25,0,0),
    # 0.25*Bt = (0.225,0.1,0), 0.25*Ct = (0,0,0.25): half-extents are
    # (0.2375, 0.05, 0.125). Sample bbox is x in [0,1.425], y in [0,0.3],
    # z in [0,0.75] since x = u + 0.9v, y = 0.4v, z = w for u,v,w in
    # {0,0.25,0.5,0.75}. A transposed implementation would instead give
    # half-extents (0.125, 0.1625, 0.125), failing axes 1 and 2 here.
    lo_k, hi_k, l_k = BI.source_box(vsk)
    @test isapprox(lo_k[1], -0.2375; atol = 1e-14)
    @test isapprox(lo_k[2], -0.05;   atol = 1e-14)
    @test isapprox(lo_k[3], -0.125;  atol = 1e-14)
    @test isapprox(hi_k[1],  1.6625; atol = 1e-14)
    @test isapprox(hi_k[2],  0.35;   atol = 1e-14)
    @test isapprox(hi_k[3],  0.875;  atol = 1e-14)
    @test isapprox(l_k[1],   1.9;    atol = 1e-14)
    @test isapprox(l_k[2],   0.4;    atol = 1e-14)
    @test isapprox(l_k[3],   1.0;    atol = 1e-14)
end

@testset "lattice_spacing falls back for non-lattice sources" begin
    # Cubic point cloud with no basis: fallback must equal the grid spacing,
    # which is what guarantees cubic-lattice results are unchanged.
    n = 6; h = 0.25
    pts = Matrix{Float64}(undef, 3, n^3); m = 0
    for k in 1:n, j in 1:n, i in 1:n
        m += 1
        pts[1, m] = i * h; pts[2, m] = j * h; pts[3, m] = k * h
    end
    vsf = VolumeSource(pts, fill(h^3, n^3), fill(1.0, n^3))
    @test vsf.lattice_basis === nothing
    @test isapprox(BI.lattice_spacing(vsf), h; rtol = 1e-12)
end

@testset "_classify_near_far_targets uses the Eq. (3.9) box" begin
    gsrc = BI.GaussianVolumeSource((0.0, 0.0, 0.0), 0.3, 12, 1e-6)
    g = BI.near_field_geometry(gsrc; c_pad = 5.0)

    # A corner of B_pad is inside the box but far from every source point, so the
    # old KDTree ball of radius 5h classified it FAR while Eq. (3.9) calls it NEAR.
    corner = [g.hi[1] - 1e-9; g.hi[2] - 1e-9; g.hi[3] - 1e-9]
    inside = [0.0; 0.0; 0.0]
    outside = [g.hi[1] + g.hn; 0.0; 0.0]
    targets = hcat(corner, inside, outside)

    is_near = BI._classify_near_far_targets(targets, gsrc; c_pad = 5.0)
    @test is_near == [true, true, false]

    # agrees with in_near_region on every column, by construction
    for i in 1:size(targets, 2)
        @test is_near[i] == BI.in_near_region(g, targets, i)
    end
end

@testset "all near/far paths agree on classification" begin
    # One cubic and one skew lattice; every INDEPENDENTLY-written near/far
    # implementation must classify identically. Of the near/far entry points
    # that funnel through near_field_geometry (see src/core/source_geometry.jl
    # header), this testset cross-checks three that each rebuild a geometry
    # and re-run their own classification loop:
    #   - BI._classify_near_far_targets       (src/shape/box3d_fmm_helpers.jl)
    #   - BI._classify_near_far_panels        (src/shape/box3d_fmm_helpers.jl)
    #   - PrecomputedVolumeField.in_field_box  (src/shape/volume_field.jl)
    # against `ref`, computed here from a *separately* obtained geometry `g`
    # and BI.in_near_region directly. A future edit that made any one of these
    # call sites build its geometry from the wrong source, the wrong c_pad, or
    # reintroduce the old KDTree-ball classifier would make it disagree with
    # `ref` here, even though today all four paths happen to reduce to the
    # same near_field_geometry + in_near_region computation.
    #
    # Two entry points are deliberately NOT cross-checked here:
    #   - evaluate_batch_potential (src/solver/lattice_batch.jl:236-237) builds
    #     its geom inline as `near_field_geometry(...); in_near_region(...)` --
    #     textually the same two calls `ref` below makes, so comparing it here
    #     would add no discriminating power over `ref` itself. It IS exercised
    #     against physics-level TKM3D/PrecomputedVolumeField agreement in
    #     test/solver/lattice_batch.jl ("near/far split consistency", "geometry
    #     and the c_pad 2-vs-5 bound"), which is a stronger check than a
    #     classification-only comparison would be.
    #   - four_index_matrix (src/solver/multi_rhs.jl:326) constructs and
    #     evaluates a PrecomputedVolumeField directly on the screened source --
    #     it does not compute classification independently, it IS a
    #     PrecomputedVolumeField call. Comparing its classification to
    #     PrecomputedVolumeField's would compare that object to itself; a test
    #     doing so would look like coverage without being any.
    function cubic_source(n)
        h = 2.0 / n
        xs = collect(-1.0 + h/2 .+ h .* (0:n-1))
        dens = Array{Float64,3}(undef, n, n, n)
        for k in 1:n, j in 1:n, i in 1:n
            r2 = xs[i]^2 + xs[j]^2 + xs[k]^2
            dens[i,j,k] = exp(-r2 / (2 * 0.25^2))
        end
        return VolumeSource((xs, xs, xs), fill(h^3, n, n, n), dens)
    end

    function skew_source(n)
        frac = collect((i - 1) / n for i in 1:n)
        At = (2.0, 0.0, 0.0); Bt = (0.7, 1.9, 0.0); Ct = (0.0, 0.0, 2.2)
        dens = fill(1.0, n, n, n)
        jac = abs(det([2.0 0.7 0.0; 0.0 1.9 0.0; 0.0 0.0 2.2]))
        return VolumeSource((frac, frac, frac), fill(jac / n^3, n, n, n), dens,
                            (0.0, 0.0, 0.0), (At, Bt, Ct))
    end

    for vs in (cubic_source(12), skew_source(12))
        g = BI.near_field_geometry(vs; c_pad = 5.0)
        # a spread of targets straddling the B_pad boundary in every direction;
        # f = 0.999/1.001 sit just inside/outside dB_pad and are what would catch
        # an off-by-half-a-cell error in source_box if one path used it and
        # another didn't.
        pts = Float64[]
        for sx in (-1.0, 0.0, 1.0), sy in (-1.0, 0.0, 1.0), sz in (-1.0, 0.0, 1.0)
            for f in (0.5, 0.999, 1.001, 2.0)
                append!(pts, [g.center[1] + sx * f * (g.hi[1] - g.center[1]),
                              g.center[2] + sy * f * (g.hi[2] - g.center[2]),
                              g.center[3] + sz * f * (g.hi[3] - g.center[3])])
            end
        end
        targets = reshape(pts, 3, :)

        ref = [BI.in_near_region(g, targets, i) for i in 1:size(targets, 2)]

        # path: box3d target classifier
        @test BI._classify_near_far_targets(targets, vs; c_pad = 5.0) == ref

        # path: box3d panel classifier. _classify_near_far_panels first reduces
        # each panel to a centroid (_panel_representative_points) and THEN
        # applies the same near_field_geometry + in_near_region computation, so
        # build degenerate panels whose centroid is exactly each target column
        # (a = 4*target, b = c = d = 0, so (a+b+c+d)/4 == target bit-exactly --
        # no floating-point reconstruction noise near the 0.999/1.001 boundary
        # points) to exercise that centroid-extraction step, which
        # _classify_near_far_targets does not go through at all.
        panels = [BI.TempPanel3D(ntuple(k -> 4 * targets[k, i], 3),
                                  (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                                  false, false, false, false,
                                  false, false, false, false,
                                  (0.0, 0.0, 1.0))
                   for i in 1:size(targets, 2)]
        @test BI._classify_near_far_panels(panels, vs; c_pad = 5.0) == ref

        # path: PrecomputedVolumeField
        f = PrecomputedVolumeField(vs; tol = 1e-6, compute_grad = false, c_pad = 5.0)
        @test [BI.in_field_box(f, targets, i) for i in 1:size(targets, 2)] == ref
        # and its stored geometry is the same numbers, independently computed
        @test f.geom.lo == g.lo && f.geom.hi == g.hi
        @test f.geom.L == g.L && f.geom.dks == g.dks
    end
end

@testset "cubic lattice: ||A_rho||_2 equals the old spacing estimate" begin
    # Guarantees that results on cubic grids are unchanged by the h redefinition
    # (Eq. 3.3's h = ||A_rho||_2 vs. the pre-Task-1 minimum-nearest-neighbour
    # estimate): on a cubic grid the two must agree exactly, so lattice_spacing
    # cannot silently move c_pad-derived quantities for the common case.
    for n in (8, 13, 24)
        h = 2.0 / n
        xs = collect(-1.0 + h/2 .+ h .* (0:n-1))
        vs = VolumeSource((xs, xs, xs), fill(h^3, n, n, n), fill(1.0, n, n, n))
        @test isapprox(BI.lattice_spacing(vs), BI._estimate_source_spacing(vs); rtol = 1e-12)
    end
end
