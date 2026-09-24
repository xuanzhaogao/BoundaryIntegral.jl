function _ltkm3dd_spreadonly_upsampfac(nmodes::NTuple{3, Int}, eps::Real)
    return 1.00001
end

function _ltkm3dd_spreadonly_next235even(n::Integer)
    n <= 2 && return 2
    nplus = isodd(n) ? n - 1 : n - 2
    numdiv = 2
    while numdiv > 1
        nplus += 2
        numdiv = nplus
        while numdiv % 2 == 0
            numdiv ÷= 2
        end
        while numdiv % 3 == 0
            numdiv ÷= 3
        end
        while numdiv % 5 == 0
            numdiv ÷= 5
        end
    end
    return nplus
end

function _ltkm3dd_spreadonly_leg_eval(n::Int, x::Float64)
    if n == 0
        return 1.0, 0.0
    elseif n == 1
        return x, 1.0
    end
    p0 = 0.0
    p1 = 1.0
    p2 = x
    for i in 1:(n - 1)
        p0 = p1
        p1 = p2
        p2 = ((2 * i + 1) * x * p1 - i * p0) / (i + 1)
    end
    return p2, n * (x * p2 - p1) / (x^2 - 1.0)
end

function _ltkm3dd_spreadonly_gaussquad(n::Int)
    xgl = Vector{Float64}(undef, n)
    wgl = Vector{Float64}(undef, n)
    xgl[(n ÷ 2) + 1] = 0.0
    for i in 0:(n ÷ 2 - 1)
        x = cos((2 * i + 1) * π / (2 * n))
        convcount = 0
        iter = 0
        max_iter = 100
        while true
            p, dp = _ltkm3dd_spreadonly_leg_eval(n, x)
            dx = -p / dp
            x += dx
            convcount = abs(dx) < 1e-14 ? convcount + 1 : 0
            convcount == 3 && break
            iter += 1
            iter >= max_iter && error("_ltkm3dd_spreadonly_gaussquad failed to converge for n=$(n), i=$(i)")
        end
        xgl[i + 1] = -x
        xgl[n - i] = x
    end
    for i in 1:(n ÷ 2 + 1)
        _, dp = _ltkm3dd_spreadonly_leg_eval(n, xgl[i])
        p, _ = _ltkm3dd_spreadonly_leg_eval(n + 1, xgl[i])
        wgl[i] = -2.0 / ((n + 1) * dp * p)
        wgl[n - i + 1] = wgl[i]
    end
    return xgl, wgl
end

function _ltkm3dd_spreadonly_kernel_params(tol::Real, sigma::Real; kerformula::Int = 1)
    sigma > 1 || throw(ArgumentError("spread-only upsampfac must be greater than 1"))
    tol_eff = max(Float64(tol), eps(Float64))

    kerformula == 1 || throw(ArgumentError("only ES spread_kerformula = 1 is supported"))

    if sigma == 2.0
        ns = ceil(Int, log10(10.0 / tol_eff))
    else
        ns = ceil(Int, log(1.0 / tol_eff) / (π * sqrt(1.0 - 1.0 / sigma)))
    end
    ns = clamp(ns, 2, 16)

    betaoverns = ns == 2 ? 2.20 : ns == 3 ? 2.26 : ns == 4 ? 2.38 : 2.30
    beta = betaoverns * ns
    if sigma != 2.0
        beta = 0.97 * π * ns * (1.0 - 1.0 / (2.0 * sigma))
    end

    return (nspread = ns, upsampfac = Float64(sigma), beta = beta, kerformula = kerformula)
end

function _ltkm3dd_spreadonly_kernel_definition(params, z::Float64)
    abs(z) > 1.0 && return 0.0
    arg = params.beta * sqrt(max(0.0, 1.0 - z^2))
    return exp(arg - params.beta)
end

function _ltkm3dd_spreadonly_poly_fit(f, n::Int)
    t = Vector{Float64}(undef, n)
    y = Vector{Float64}(undef, n)
    for k in 0:(n - 1)
        t[k + 1] = cos((2 * k + 1) * π / (2 * n))
        y[k + 1] = f(t[k + 1])
    end

    coef = copy(y)
    for j in 2:n
        for i in n:-1:j
            coef[i] = (coef[i] - coef[i - 1]) / (t[i] - t[i - j + 1])
        end
    end

    function mul_by_linear(p::Vector{Float64}, c::Float64)
        r = zeros(Float64, length(p) + 1)
        for i in eachindex(p)
            r[i] += -c * p[i]
            r[i + 1] += p[i]
        end
        return r
    end

    c = zeros(Float64, n)
    basis = [1.0]
    c[1] += coef[1]
    for j in 2:n
        basis = mul_by_linear(basis, t[j - 1])
        for m in eachindex(basis)
            c[m] += coef[j] * basis[m]
        end
    end
    reverse!(c)
    return c
end

function _ltkm3dd_spreadonly_horner_coeffs(params)
    ns = params.nspread
    nc_fit = min(19, ns + 3)
    coeffs = Matrix{Float64}(undef, nc_fit, ns)
    for j in 1:ns
        xshiftj = 2 * (j - 1) + 1 - ns
        kernel_this_interval = x -> begin
            z = (x + xshiftj) / ns
            _ltkm3dd_spreadonly_kernel_definition(params, z)
        end
        coeffs[:, j] .= _ltkm3dd_spreadonly_poly_fit(kernel_this_interval, nc_fit)
    end
    return coeffs
end

function _ltkm3dd_spreadonly_evaluate_kernel_runtime(x::Float64, coeffs::AbstractMatrix{<:Real}, ns::Int)
    ns2 = ns / 2.0
    res = 0.0
    for i in 1:ns
        if x > -ns2 + (i - 1) && x <= -ns2 + i
            z = muladd(2.0, x - (i - 1), ns - 1)
            for j in axes(coeffs, 1)
                res = muladd(res, z, coeffs[j, i])
            end
            break
        end
    end
    return res
end

function _ltkm3dd_spreadonly_onedim_fseries_kernel(nf::Int, params, coeffs)
    J2 = params.nspread / 2.0
    q = Int(2 + 3.0 * J2)
    z, w = _ltkm3dd_spreadonly_gaussquad(2 * q)
    f = Vector{Float64}(undef, q)
    a = Vector{ComplexF64}(undef, q)
    for n in 1:q
        zn = z[n] * J2
        f[n] = J2 * w[n] * _ltkm3dd_spreadonly_evaluate_kernel_runtime(zn, coeffs, params.nspread)
        a[n] = -exp(2π * im * zn / nf)
    end

    aj = ones(ComplexF64, q)
    fwkerhalf = Vector{Float64}(undef, nf ÷ 2 + 1)
    for j in 0:(nf ÷ 2)
        x = 0.0
        for n in 1:q
            x += f[n] * 2.0 * real(aj[n])
        end
        fwkerhalf[j + 1] = x
        for n in 1:q
            aj[n] *= a[n]
        end
    end
    return fwkerhalf
end

@inline function _ltkm3dd_spreadonly_fwindex(k::Int, nf::Int)
    return k >= 0 ? k + 1 : nf + k + 1
end

@inline function _ltkm3dd_spreadonly_fkindex(k::Int, m::Int)
    return k + (m ÷ 2) + 1
end

function _ltkm3dd_spreadonly_deconvolveshuffle3d_dir2(
    fk::AbstractArray{Complex{T}, 3},
    nfdim::NTuple{3, Int},
    phi1::AbstractVector{T},
    phi2::AbstractVector{T},
    phi3::AbstractVector{T},
) where {T <: AbstractFloat}
    ms, mt, mu = size(fk)
    nf1, nf2, nf3 = nfdim
    fw = zeros(Complex{T}, nf1, nf2, nf3)
    k1min, k1max = -(ms ÷ 2), (ms - 1) ÷ 2
    k2min, k2max = -(mt ÷ 2), (mt - 1) ÷ 2
    k3min, k3max = -(mu ÷ 2), (mu - 1) ÷ 2
    @inbounds for k3 in k3min:k3max, k2 in k2min:k2max, k1 in k1min:k1max
        fw[_ltkm3dd_spreadonly_fwindex(k1, nf1), _ltkm3dd_spreadonly_fwindex(k2, nf2), _ltkm3dd_spreadonly_fwindex(k3, nf3)] =
            fk[_ltkm3dd_spreadonly_fkindex(k1, ms), _ltkm3dd_spreadonly_fkindex(k2, mt), _ltkm3dd_spreadonly_fkindex(k3, mu)] /
            (phi1[abs(k1) + 1] * phi2[abs(k2) + 1] * phi3[abs(k3) + 1])
    end
    return fw
end
