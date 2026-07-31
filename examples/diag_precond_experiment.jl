# Does diagonal (contrast) preconditioning of the dielectric BIE help?
#
# The LHS is A = G + Dᵀ + Cᵀ with G = diag(t_P) block-diagonal and constant per interface
# (t_P = ½(ε_out+ε_in)/(ε_out−ε_in) = −½γ_P in the paper's sign convention). Unlike a
# textbook second-kind equation, the identity term is not ½I: A's spectrum sits in one
# cluster per distinct contrast, and in a multi-material geometry those clusters straddle
# the origin. Right-preconditioning with P = G⁻¹ collapses them to a single cluster at 1.
#
# Predictions this script is meant to falsify:
#   1. single box in vacuum  -> IDENTICAL iteration count (P is a scalar multiple of I and
#      GMRES is invariant under scalar scaling). Doubles as a correctness check.
#   2. three-material §6.1 geometry -> a modest drop (γ spans −11 … +2.3 there).
#   3. growth of N_iter with ε₂ (the |γ|→1 conductor limit) -> essentially UNCHANGED slope,
#      because that growth is ‖(Dᵀ+Cᵀ)P‖ → 1, which no diagonal can fix.
#
# Usage (quick, a workstation; prints N so you can see the problem size):
#   julia --project -t 16 examples/diag_precond_experiment.jl
#
# Paper-scale (§6.1 uses p=6, τ=1e-4); tighten via env vars, e.g. on one node:
#   BI_PC_NQUAD=6 BI_PC_RHS_TOL=1e-4 BI_PC_LEC=0.3125 BI_PC_MAXDEPTH=16 \
#   OMP_NUM_THREADS=96 OMP_PROC_BIND=spread OPENBLAS_NUM_THREADS=1 \
#   julia --project -t 96 examples/diag_precond_experiment.jl
#
# Only N_iter is being compared, so the two runs of a case always share one interface and
# one RHS; nothing but the preconditioner changes between them.

using BoundaryIntegral
using LinearAlgebra
using Printf

const NQUAD       = parse(Int,     get(ENV, "BI_PC_NQUAD",       "4"))
const RHS_TOL     = parse(Float64, get(ENV, "BI_PC_RHS_TOL",     "1e-2"))
const LEC         = parse(Float64, get(ENV, "BI_PC_LEC",         "2.5"))
const LHS_TOL     = parse(Float64, get(ENV, "BI_PC_LHS_TOL",     "1e-6"))
const GMRES_RTOL  = parse(Float64, get(ENV, "BI_PC_GMRES_RTOL",  "1e-8"))
const MAXDEPTH    = parse(Int,     get(ENV, "BI_PC_MAXDEPTH",    "8"))
const MAX_ORDER   = parse(Int,     get(ENV, "BI_PC_MAX_ORDER",   "8"))
const ITMAX       = parse(Int,     get(ENV, "BI_PC_ITMAX",       "1000"))
const SRC_WIDTH   = parse(Float64, get(ENV, "BI_PC_SRC_WIDTH",   "0.3"))
const SRC_N       = parse(Int,     get(ENV, "BI_PC_SRC_N",       "9"))
# With the default (false), rim panels get NO near correction and the conditioning of A
# degrades as the edges are refined — so N_it is NOT mesh-independent. Set to 1 to measure
# the properly corrected operator.
const CORRECT_EDGES = get(ENV, "BI_PC_CORRECT_EDGES", "0") in ("1", "true")

"Normalized Gaussian of width `width` on a uniform (SRC_N)³ grid spanning ±3σ."
function gaussian_source(center::NTuple{3, Float64}, width::Float64;
        n::Int = SRC_N, halfw::Float64 = 3.0)
    r = halfw * width
    ax(c) = collect(range(c - r, c + r, length = n))
    xs, ys, zs = ax(center[1]), ax(center[2]), ax(center[3])
    h = xs[2] - xs[1]
    pts = [(x, y, z) for x in xs for y in ys for z in zs]
    pos = reduce(hcat, [collect(p) for p in pts])
    w = fill(h^3, length(pts))
    dens = [exp(-((p[1] - center[1])^2 + (p[2] - center[2])^2 + (p[3] - center[3])^2) /
                (2 * width^2)) for p in pts]
    dens ./= sum(dens .* w)                     # ∫ρ = 1
    return VolumeSource(pos, w, dens)
end

"Distinct paper-convention contrasts γ = (ε_in+ε_out)/(ε_in−ε_out) present in the mesh."
function contrasts(interface)
    gs = Float64[]
    for i in 1:length(interface.panels)
        ei, eo = interface.eps_in[i], interface.eps_out[i]
        g = (ei + eo) / (ei - eo)
        any(x -> isapprox(x, g; rtol = 1e-12), gs) || push!(gs, g)
    end
    return sort!(gs)
end

function run_case(label, boxes, epses, src_center; eps_out = 1.0)
    vs = gaussian_source(src_center, SRC_WIDTH)
    interface = multi_dielectric_box3d_rhs_adaptive(NQUAD, LEC, boxes, epses, [vs], RHS_TOL;
        eps_out = eps_out, max_depth = MAXDEPTH)
    N = BoundaryIntegral.num_points(interface)
    gs = contrasts(interface)
    s = dielectric_diagonal_scaling(interface)
    # cond(G) as seen by GMRES: the spread of the diagonal clusters.
    spread = maximum(abs, gs) / minimum(abs, gs)

    @info "case $label" N n_panels=length(interface.panels) gammas=gs gamma_spread=spread

    out = Dict{Bool, NamedTuple}()
    for pc in (false, true)
        t = @elapsed ((Σ, stats) = solve_dielectric_box3d_block(interface, [vs];
            fmm_tol = LHS_TOL, up_tol = LHS_TOL, max_order = MAX_ORDER,
            rtol = GMRES_RTOL, itmax = ITMAX, precondition = pc,
            correct_edges = CORRECT_EDGES,
            screen_boxes = boxes, screen_epses = epses, screen_eps_out = eps_out))
        out[pc] = (; niter = stats.niter, solved = stats.solved, time = t, sigma = Σ[:, 1])
    end

    σ0, σ1 = out[false].sigma, out[true].sigma
    dσ = norm(σ1 .- σ0) / norm(σ0)
    return (; label, N, gammas = gs, spread,
        niter_plain = out[false].niter, niter_pc = out[true].niter,
        t_plain = out[false].time, t_pc = out[true].time,
        solved = out[false].solved && out[true].solved,
        sigma_reldiff = dσ,
        scaling_range = (minimum(s), maximum(s)))
end

# --- Case 1: single box in vacuum. One γ ⇒ P ∝ I ⇒ must be a no-op. ------------------
single = run_case("single box (ε=10, one γ)",
    [(center = (0.0, 0.0, 0.0), Lx = 4.0, Ly = 4.0, Lz = 4.0)], [10.0], (0.0, 0.0, 0.0))

# --- Case 2: the §6.1 three-material geometry, sweeping ε₂ --------------------------
# 10×10×1 slab (ε₃=10) resting on two 10×10×10 cubes (ε₁=4, ε₂ swept) that share the
# internal face x=0, all in vacuum; the source sits inside the slab, off the junction.
const THREE_MAT_BOXES = [
    (center = (-5.0, 0.0, -5.0), Lx = 10.0, Ly = 10.0, Lz = 10.0),   # Ω₁, ε₁ = 4
    (center = ( 5.0, 0.0, -5.0), Lx = 10.0, Ly = 10.0, Lz = 10.0),   # Ω₂, ε₂ swept
    (center = ( 0.0, 0.0,  0.5), Lx = 10.0, Ly = 10.0, Lz =  1.0),   # Ω₃, ε₃ = 10 (host)
]
const SRC_IN_SLAB = (2.0, 0.0, 0.5)

eps2_list = [parse(Float64, x) for x in split(get(ENV, "BI_PC_EPS2", "6,20,60,200"), ',')]
sweep = [run_case(@sprintf("three-material, ε₂=%g", e2),
                  THREE_MAT_BOXES, [4.0, e2, 10.0], SRC_IN_SLAB) for e2 in eps2_list]

# --- Report -------------------------------------------------------------------------
println()
println("p = $NQUAD, τ_rhs = $RHS_TOL, τ_lhs = $LHS_TOL, l_ec = $LEC, ",
        "gmres_rtol = $GMRES_RTOL, correct_edges = $CORRECT_EDGES")
println()
@printf("%-28s %9s %8s %8s %8s %9s %9s %10s\n",
    "case", "N", "|γ|spread", "N_it", "N_it(P)", "t (s)", "t(P) (s)", "‖Δσ‖/‖σ‖")
println(repeat("-", 100))
for r in vcat(single, sweep)
    @printf("%-28s %9d %8.2f %8d %8d %9.1f %9.1f %10.2e%s\n",
        r.label, r.N, r.spread, r.niter_plain, r.niter_pc, r.t_plain, r.t_pc,
        r.sigma_reldiff, r.solved ? "" : "  [NOT CONVERGED]")
end
println()
println("γ values per case:")
for r in vcat(single, sweep)
    @printf("  %-28s %s   (scaling 1/t ∈ [%.3f, %.3f])\n",
        r.label, string(round.(r.gammas; digits = 3)),
        r.scaling_range[1], r.scaling_range[2])
end
println()
if single.niter_plain == single.niter_pc
    println("OK  single-box invariance holds (N_it identical), as it must for one γ.")
else
    println("FAIL  single-box N_it changed ($(single.niter_plain) -> $(single.niter_pc)); ",
            "with one γ the preconditioner is a scalar and GMRES is scale-invariant, ",
            "so this indicates a bug in the scaling, not a real effect.")
end
