# Bernstein-radius utilities for the near-field criterion.
#
# For a panel pair we follow af Klinteberg & Tornberg (2022): the squared
# distance R²(s; x) = ‖y(s) - x‖² between a target point x and the source
# panel (mapped to reference coordinates s ∈ [-1,1]²) has, for each fixed
# reference line, a complex-conjugate pair of poles off the real axis. The
# Bernstein radius ρ(t₀) of the nearest pole controls the p-point
# Gauss–Legendre error as O(ρ^{-2p}); the slower of the two axis rates wins.

# Bernstein radius of a single pole t₀ (the focal-sum length of the largest
# Bernstein ellipse with foci ±1 on which the kernel stays analytic, ρ ≥ 1).
function bernstein_rho_from_pole(z::Complex{T}) where T <: Real
    s  = sqrt(z * z - 1)        # complex sqrt (principal branch)
    w1 = z + s
    w2 = z - s
    return max(abs(w1), abs(w2))
end

# Panel-pair Bernstein radius for a single target point.
#
# `panel` is the source panel Q being integrated; `trg` is one target node
# x ∈ P. We map x into Q's local frame (centre cc, unit in-plane axes ê₁ ∝ b-a
# and ê₂ ∝ d-a, normal n̂) with half-lengths L₁, L₂, giving local coordinates
# (xt, yt, zt). The two reference-axis poles are
#
#   t₀⁽¹⁾(s₂) = (xt + i·√((yt - η₂)² + zt²)) / L₁,   η₂ = L₂·s₂, s₂ ∈ [-1,1]
#   t₀⁽²⁾(s₁) = (yt + i·√((xt - η₁)² + zt²)) / L₂,   η₁ = L₁·s₁, s₁ ∈ [-1,1]
#
# The inner min over s of ρ(t₀) (Eq. (rho_min) of the manuscript) is reached
# analytically: at fixed real part, ρ is monotone increasing in |Im t₀|, so the
# minimizing node is η = clamp of the in-plane target coordinate onto the panel.
# We return min(ρ⁽¹⁾, ρ⁽²⁾) — the slower (smaller-ρ) axis dictates the rate.
function bernstein_rho_panel(panel::FlatPanel{T, 3}, trg::NTuple{3, T}) where T
    a, b, c, d = panel.corners
    cc = (a .+ b .+ c .+ d) ./ 4

    u1 = b .- a
    u2 = d .- a
    len1 = norm(u1)
    len2 = norm(u2)
    L1 = len1 / 2
    L2 = len2 / 2
    e1 = u1 ./ len1
    e2 = u2 ./ len2

    r  = trg .- cc
    xt = dot(r, e1)
    yt = dot(r, e2)
    zt = dot(r, panel.normal)

    # axis-1 pole: minimise |Im| over η₂ ∈ [-L₂, L₂] (closest reference node)
    dy   = yt - clamp(yt, -L2, L2)
    t1   = Complex{T}(xt, sqrt(dy * dy + zt * zt)) / L1
    rho1 = bernstein_rho_from_pole(t1)

    # axis-2 pole: minimise |Im| over η₁ ∈ [-L₁, L₁]
    dx   = xt - clamp(xt, -L1, L1)
    t2   = Complex{T}(yt, sqrt(dx * dx + zt * zt)) / L2
    rho2 = bernstein_rho_from_pole(t2)

    return min(rho1, rho2)
end
