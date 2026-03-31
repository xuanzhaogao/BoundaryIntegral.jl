# Predict corner singularity powers for 2D dielectric wedge
# using idea as in van Bladel's EM book, Sec 4.13.
# Barnett 11/6/25
#
# The "interior" material with relative permittivity epsilon
# occupies angle alpha. Exterior epsilon=1. Matching is
#     epsilon phi_n^- = phi_n^+
# where + denotes exterior and - interior side.
#
# For our SLP representation, the density power is the jump in normal
# derivative.
function theta_shooting_even(theta::T, eps::T, g::T) where T
	return sin(g * theta / 2) * cos(g * (π - theta / 2)) + cos(g * theta / 2) * sin(g * (π - theta/2)) / eps
end

function theta_shooting_odd(theta::T, eps::T, g::T) where T
    return cos(g * theta / 2) * sin(g * (π - theta / 2)) + sin(g * theta / 2) * cos(g * (π - theta / 2)) / eps
end

# Predict corner singularity power sequence for multijunction 2D dielectric
# generalized wedge, using idea as in van Bladel's EM book, Sec 4.13.
# Barnett 11/7/25, generalizing cornervanbladel2d.jl to use 2x2 det of ODE
# transmission matrix (or one element of such matrix when PEC).
#
# There are nm>=2 materials including vacuum (nm=2 is a plain diel corner).
# The last material is optionally PEC (eps=Inf).
# The relative permittivities are length-nm vector e, and angles a length nm-1
# vector (the last angle defined implicitly).
# Matching at each junction is
#     eps_j phi_n^- = eps_{j+1} phi_n^+
# where -(+) denotes  theta just below (above).
# For our SLP representation, the density power is the jump in normal
# derivative.
"""
	theta_ODE_det(a,e,g)

	return determinant for theta ODE transmission for multi-junction dielectric
	relative premittivity vector `e` and angles `a` (length one less than `e`,
	since the last angle is known), and trial power `g`
	(so u''+g^2u = 0 is the theta ODE in each material).
	Zero is returned iff `g` is a (nonlin) eigenvalue of the periodic ODE on
	[0,2pi].
	The last `e` value (only) may be `Inf`, in which case Dirichlet BCs are used
	on its angle endpoints (and the incoming u' to final u value map used).
	Note: Parity issues avoided due to solving the 2x2 transmission matrix
	on the entire [0,2pi] domain. (All parities are combined.)
"""
function theta_ODE_det(a::AbstractVector, e::AbstractVector, g)
    M = Matrix(1.0I, 2, 2)     # initial I transm matrix (at theta=0^+)
    aa = [a; 2pi - sum(a)]       # add the last angle
    @assert aa[end] >= 0 "angles ($a) sum to bigger than 2pi!"
    nm = length(e)           # no. materials
    @assert length(aa) == nm "input vectors wrong lengths!"
    nexte = circshift(e, -1)   # list of j+1'th epsilons
    for (j, e) in enumerate(e)
        if !isinf(e)          # if last material conductor (u=0) skip it
            c = cos(g * aa[j])
            s = sin(g * aa[j])      # where nonlin in g enters
            # update transm matrix by propagate by angle a then scale u'...
            soverg = g == 0.0 ? aa[j] : s / g       # handle g=0 (u=at+b)
            M = [1.0 0; 0 e/nexte[j]] * [c soverg; -g*s c] * M
        end
    end
    return isinf(e[end]) ? M[1, 2] : det(M - I)   # use u' -> u map if Dir BCs
end

"""
    corner_singularity_power(alpha, eps_in, eps_out; g0=1.0)

Compute the leading singularity power `gamma` for a two-material dielectric
wedge corner. The potential behaves as `r^gamma` near the corner, and the
SLP density as `s^{gamma-1}`.

# Arguments
- `alpha`: interior wedge angle (angle occupied by material `eps_in`)
- `eps_in`: permittivity of the interior material
- `eps_out`: permittivity of the exterior material
- `g0`: initial guess for the root finder (default 1.0)

# Returns
The leading singularity power `gamma`. When `eps_in == eps_out`, returns 1.0
exactly (no singularity). When `gamma < 1`, the density is singular at the corner.
"""
function corner_singularity_power(alpha::Real, eps_in::Real, eps_out::Real; g0=1.0)
    if eps_in == eps_out
        return 1.0
    end
    # The shooting functions use the convention that `theta` is the wedge angle
    # of the material with permittivity ratio `eps` (relative to the other material).
    # Here `alpha` is the angle occupied by `eps_in`. We pass `theta = alpha` and
    # `eps = eps_out / eps_in` so that the complementary material (eps_in) effectively
    # plays the role of the "exterior" in the shooting equation.
    # The even-parity root is the leading singularity power for the potential.
    eps_ratio = eps_out / eps_in
    g_even = fzero(g -> theta_shooting_even(Float64(alpha), Float64(eps_ratio), g), Float64(g0))
    return g_even
end

"""
    corner_singularity_power_multi(angles, epsilons; g0=1.0)

Compute the leading singularity power `gamma` for a multi-junction dielectric
corner using `theta_ODE_det`.

# Arguments
- `angles`: vector of wedge angles (length `nm-1` where `nm` is number of materials;
  the last angle is `2pi - sum(angles)`)
- `epsilons`: vector of relative permittivities (length `nm`)
- `g0`: initial guess for the root finder (default 1.0)

# Returns
The leading singularity power `gamma`.
"""
function corner_singularity_power_multi(angles::AbstractVector, epsilons::AbstractVector; g0=1.0)
    gamma = fzero(g -> theta_ODE_det(Float64.(angles), Float64.(epsilons), g), Float64(g0))
    return gamma
end