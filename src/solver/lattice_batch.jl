# src/solver/lattice_batch.jl
# Lattice-scale batches: frame-TRANSLATED orbital instances on a virtual global grid
# (spec: docs/superpowers/specs/2026-06-10-multinode-lattice-campaign-design.md).
# Unlike the .bie LATTICE images (periodic circshift on the template grid), instances
# here carry an integer global-frame offset and never wrap — so >5 distinct cells per
# direction are representable and pair products are exact on frame intersections.

"""
    lattice_grid_steps(datagrid, primvec, n::NTuple{3,Int}) -> NTuple{3,Int}

Integer grid-step offset of the lattice translation `n1·a1 + n2·a2 + n3·a3` (the
global-frame offset of a translated orbital instance). Errors if a lattice vector is
not grid-commensurate. Same arithmetic as the `.bie` LATTICE images, but interpreted
as a frame translation, not a circshift.
"""
lattice_grid_steps(datagrid, primvec::AbstractMatrix, n::NTuple{3,Int}) =
    _lattice_grid_shift(datagrid, primvec, n)

"""
    OrbitalInstance(id, template_id, steps)

One orbital of a lattice campaign: template `template_id` translated by the integer
global-frame offset `steps` (grid steps; see `lattice_grid_steps`).
"""
struct OrbitalInstance
    id::Int
    template_id::Int
    steps::NTuple{3,Int}
end

# Per-axis overlap of two frames of length n offset by integer steps si, sj.
# Returns the GLOBAL index range covered by both (local_i = g - si, local_j = g - sj),
# or `nothing` if disjoint.
function _frame_overlap(n::Int, si::Int, sj::Int)
    glo = max(si, sj) + 1
    ghi = min(si, sj) + n
    glo > ghi && return nothing
    return glo:ghi
end
