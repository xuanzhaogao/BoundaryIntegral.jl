# Snap an arbitrary Cartesian orbital position to the nearest integer grid offset on the
# template's grid, preserving the exact commensurate-grid machinery (spec §4).

using LinearAlgebra

"""
    snap_orbital(datagrid, centroid, pos) -> NTuple{3,Int}

Integer grid-step offset that places an orbital (template `datagrid`, density centroid
`centroid`) closest to Cartesian `pos`. Solves `pos - centroid = G * s` for real `s` in
the grid-step basis `G = [At/nx Bt/ny Ct/nz]` (handles non-orthogonal a1/a2), then rounds.
Warns if the snap moves the center by more than half the largest grid step; errors if the
z-offset is not (near-)integer (a planar campaign — a fractional z is almost certainly a
frame/unit mistake).
"""
function snap_orbital(datagrid, centroid::NTuple{3,Float64}, pos::NTuple{3,Float64})
    At, Bt, Ct = true_cell_vectors(datagrid)
    G = hcat(collect(At) ./ datagrid.nx, collect(Bt) ./ datagrid.ny, collect(Ct) ./ datagrid.nz)
    rhs = collect(pos) .- collect(centroid)
    s = G \ rhs                                  # real fractional steps
    steps = round.(Int, s)
    realized = collect(centroid) .+ G * steps
    max_step = maximum(norm.((G[:, 1], G[:, 2], G[:, 3])))
    if norm(rhs .- G * steps) > 0.5 * max_step
        @warn "snap_orbital: orbital snapped >½ grid step" requested=pos realized=Tuple(realized)
    end
    abs(s[3] - steps[3]) <= 1e-3 ||
        error("snap_orbital: z=$(pos[3]) is not representable on the template grid (Δ=$(s[3]-steps[3]) steps)")
    return (steps[1], steps[2], steps[3])
end
