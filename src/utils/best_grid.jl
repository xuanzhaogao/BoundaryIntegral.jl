"""
Return (m, n) for grid tiling of a W×H rectangle into m×n equal rectangles
such that each small rectangle's aspect ratio <= alpha and m*n is as small
as found by continued-fraction convergents + semiconvergents.

Assumes W>0, H>0, alpha>=1.
"""
function best_grid(W::Real, H::Real, alpha::Real;
                      max_den::Int = 1_000_000, max_iter::Int = 10_000,
                      frac_tol::Real = 1e-12)

    W = float(W); H = float(H); alpha = float(alpha)
    if !(W > 0 && H > 0)
        throw(ArgumentError("W and H must be positive"))
    end
    if alpha < 1
        throw(ArgumentError("alpha must be >= 1"))
    end

    # Ensure r >= 1 by swapping if needed, then swap back at the end.
    swapped = false
    if W < H
        W, H = H, W
        swapped = true
    end
    r = W / H  # >= 1

    # Helper: aspect ratio of each small rectangle for grid (m,n)
    # small w = W/m, h = H/n, so w/h = (W*n)/(H*m)
    aspect(m::Int, n::Int) = begin
        ratio = (W * n) / (H * m)
        ratio >= 1 ? ratio : 1 / ratio
    end

    # Track best feasible solution (minimize m*n, tie-break by closeness to 1).
    best_m, best_n = 0, 0
    best_N = typemax(Int)
    best_rho = Inf

    function consider(m::Int, n::Int)
        if m <= 0 || n <= 0
            return
        end
        # bound denominator to keep numbers sane
        if m > max_den || n > max_den
            return
        end
        ρ = aspect(m, n)
        if ρ <= alpha
            N = m * n
            if N < best_N || (N == best_N && ρ < best_rho)
                best_m, best_n = m, n
                best_N = N
                best_rho = ρ
            end
        end
        return
    end

    # Continued fraction convergents via Euclidean algorithm on real r.
    # Using standard recurrence:
    # p[-2]=0,p[-1]=1; q[-2]=1,q[-1]=0; p_k=a_k p_{k-1}+p_{k-2}, similarly q.
    pmm, pm = 0, 1
    qmm, qm = 1, 0

    x = r
    for _ in 1:max_iter
        a = floor(Int, x)
        p = a * pm + pmm
        q = a * qm + qmm

        consider(p, q)  # convergent p/q ≈ r -> use (m,n)=(p,q)

        # Also try semiconvergents between previous and current convergents:
        # (p' , q') = t*pm + pmm , t*qm + qmm for t = 1..a-1
        # These can sometimes give smaller m*n while still meeting alpha.
        if a > 1
            # Limit t to avoid huge loops when a is large due to rounding.
            t_max = a - 1
            if pm > 0
                t_max = min(t_max, (max_den - pmm) ÷ pm)
            end
            if qm > 0
                t_max = min(t_max, (max_den - qmm) ÷ qm)
            end
            for t in 1:t_max
                ps = t * pm + pmm
                qs = t * qm + qmm
                consider(ps, qs)
            end
        end

        # Stop if x is integer (no further terms).
        frac = x - a
        if abs(frac) <= frac_tol
            break
        end

        # Update recurrence
        pmm, pm = pm, p
        qmm, qm = qm, q

        x = 1 / frac
    end

    if best_m == 0
        throw(ArgumentError("No (m,n) found within search limits for alpha=$alpha"))
    end

    # If we swapped W/H, we swapped the rectangle, which swaps roles of m and n.
    if swapped
        # originally input had W<H, we swapped W<->H so r>=1.
        # For the original orientation, swap m<->n back.
        best_m, best_n = best_n, best_m
    end

    return best_m, best_n
end

"""
Alias for best_grid; kept for backward compatibility.
"""
function best_grid_mn(W::Real, H::Real, alpha::Real;
                      max_den::Int = 1_000_000, max_iter::Int = 10_000)
    return best_grid(W, H, alpha; max_den=max_den, max_iter=max_iter)
end
