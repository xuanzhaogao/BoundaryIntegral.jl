# src/core/source_geometry.jl
#
# The near/far geometry of Section 3 of the four-index BIE paper, in one place.
# All three near/far call sites (PrecomputedVolumeField, evaluate_batch_potential,
# _classify_near_far_targets) delegate here so the geometry cannot drift apart.
#
# Equation references are to ~/Articles/four_indices_bie/main.tex:
#   (3.3)  h   = ||A_rho||_2, a conservative characteristic spacing
#   (3.7)  B   = cuboidal source box, side lengths l_alpha
#   (3.10) h_n = c_pad * h
#   (3.9)  B_pad = B + [-h_n, h_n]^3; a target is near iff x in B_pad
#   (3.11) L   = sqrt(sum_alpha (l_alpha + h_n)^2), the largest source-target distance
#   (3.16) L_alpha >= l_alpha + h_n + L, i.e. dk_alpha = 2pi / (l_alpha + h_n + L)

"""
    NearFieldGeometry{T}

Section 3 near/far geometry for one volume source. `lo`/`hi` are the corners of
the padded near region `B_pad` (Eq. 3.9), `center` is the centre of `B` (which is
also the centre of `B_pad`), `l` the side lengths of `B` (Eq. 3.7), `h` the
characteristic spacing (Eq. 3.3), `hn = c_pad * h` (Eq. 3.10), `L` the truncation
radius (Eq. 3.11), and `dks` the Fourier spacings (Eq. 3.16 at equality).
"""
struct NearFieldGeometry{T}
    lo::NTuple{3, T}
    hi::NTuple{3, T}
    center::NTuple{3, T}
    l::NTuple{3, T}
    h::T
    hn::T
    L::T
    dks::NTuple{3, T}
end

"""
    lattice_spacing(vs) -> T

The characteristic spacing `h = ||A_rho||_2` of Eq. (3.3): the largest singular
value of the primitive sampling-cell basis. Falls back to
`_estimate_source_spacing` (minimum nearest-neighbour distance) when the source
carries no lattice basis; the two agree for a cubic grid.

This is the spacing for `h_n = c_pad * h` only. `k_max` keeps
`_estimate_source_spacing`, because Section 3.3 leaves `k_Nyq` open for a
general sampling lattice.
"""
function lattice_spacing(vs::VolumeSource{T, 3}) where {T}
    b = vs.lattice_basis
    b === nothing && return _estimate_source_spacing(vs)
    A = T[b[1][1] b[2][1] b[3][1];
          b[1][2] b[2][2] b[3][2];
          b[1][3] b[2][3] b[3][3]]
    s = T(opnorm(A, 2))
    return s > zero(T) ? s : _estimate_source_spacing(vs)
end

"""
    source_box(vs) -> (lo, hi, l)

The source box `B` of Eq. (3.7): the bounding box of the samples inflated by half
a lattice cell, so that the quadrature cells tile `B` and `B` contains the full
support of the piecewise-constant density. The per-axis half-extent is
`(1/2) * sum_j |A_rho[alpha, j]|`, or `h/2` when no basis is available.
"""
function source_box(vs::VolumeSource{T, 3}) where {T}
    pos = vs.positions
    size(pos, 2) >= 1 || throw(ArgumentError("source_box requires at least one source point"))
    b = vs.lattice_basis
    half = if b === nothing
        hh = _estimate_source_spacing(vs) / 2
        ntuple(_ -> hh, 3)
    else
        ntuple(a -> (abs(b[1][a]) + abs(b[2][a]) + abs(b[3][a])) / 2, 3)
    end
    lo = ntuple(a -> minimum(view(pos, a, :)) - half[a], 3)
    hi = ntuple(a -> maximum(view(pos, a, :)) + half[a], 3)
    l  = ntuple(a -> hi[a] - lo[a], 3)
    return lo, hi, l
end

"""
    near_field_geometry(vs; c_pad = 5.0) -> NearFieldGeometry

Assemble the Section 3 geometry for `vs`. `dks` uses `prevfloat` so the real-space
period is strictly greater than the Eq. (3.16) bound, keeping the trapezoidal sum
inside the aliasing-free set rather than exactly on its boundary.
"""
function near_field_geometry(vs::VolumeSource{T, 3}; c_pad::Real = 5.0) where {T}
    c_pad >= 0 || throw(ArgumentError("c_pad must be >= 0"))
    loB, hiB, l = source_box(vs)
    h = lattice_spacing(vs)
    h > zero(T) || throw(ArgumentError("lattice spacing must be positive"))
    hn = T(c_pad) * h
    lo = ntuple(d -> loB[d] - hn, 3)
    hi = ntuple(d -> hiB[d] + hn, 3)
    center = ntuple(d -> (loB[d] + hiB[d]) / 2, 3)
    L = sqrt((l[1] + hn)^2 + (l[2] + hn)^2 + (l[3] + hn)^2)
    dks = ntuple(d -> prevfloat(T(2π) / (l[d] + hn + L)), 3)
    return NearFieldGeometry{T}(lo, hi, center, l, h, hn, T(L), dks)
end

"""
    in_near_region(g, targets, i) -> Bool

Eq. (3.9): is column `i` of the `3 × n` matrix `targets` inside `B_pad`?
"""
@inline in_near_region(g::NearFieldGeometry, targets::AbstractMatrix, i::Integer) =
    (g.lo[1] <= targets[1, i] <= g.hi[1]) &&
    (g.lo[2] <= targets[2, i] <= g.hi[2]) &&
    (g.lo[3] <= targets[3, i] <= g.hi[3])
