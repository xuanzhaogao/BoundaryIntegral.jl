# defination of the laplace kernel in 2d, potential and gradient
function laplace2d_pot(src::NTuple{2, T}, trg::NTuple{2, T}) where T
    r2 = sum((src .- trg).^2)
    r = sqrt(r2)
    return log(r) / 2π
end

function laplace2d_grad(src::NTuple{2, T}, trg::NTuple{2, T}, norm::NTuple{2, T}) where T
    r2 = sum((src .- trg).^2)
    inv_r2 = one(T) / r2

    return dot(norm, inv_r2 .* (trg .- src)) / 2π
end

# filling the matrix directly, including S, D and DT
function laplace2d_S(interface::DielectricInterface{P, T}) where {P <: AbstractPanel, T}
    n_points = num_points(interface)
    S = zeros(T, n_points, n_points)
    for (i, pointi) in enumerate(eachpoint(interface))
        for (j, pointj) in enumerate(eachpoint(interface))
            i == j && continue
            @inbounds S[i, j] = laplace2d_pot(pointj.panel_point.point, pointi.panel_point.point) * pointj.panel_point.weight
        end
    end
    return S
end

function laplace2d_D(interface::DielectricInterface{P, T}) where {P <: AbstractPanel, T}
    n_points = num_points(interface)
    D = zeros(T, n_points, n_points)
    for (i, pointi) in enumerate(eachpoint(interface))
        for (j, pointj) in enumerate(eachpoint(interface))
            i == j && continue
            @inbounds D[j, i] = laplace2d_grad(pointj.panel_point.point, pointi.panel_point.point, pointi.panel_point.normal) * pointi.panel_point.weight
        end
    end
    return D
end

function laplace2d_DT(interface::DielectricInterface{P, T}) where {P <: AbstractPanel, T}
    n_points = num_points(interface)
    DT = zeros(T, n_points, n_points)
    for (i, pointi) in enumerate(eachpoint(interface))
        for (j, pointj) in enumerate(eachpoint(interface))
            i == j && continue
            @inbounds DT[i, j] = laplace2d_grad(pointj.panel_point.point, pointi.panel_point.point, pointi.panel_point.normal) * pointj.panel_point.weight
        end
    end
    return DT
end

function laplace2d_pottrg(interface::DielectricInterface{P, T}, targets::Matrix{T}) where {P <: AbstractPanel, T}
    n_points = num_points(interface)
    n_targets = size(targets, 2)
    weights = all_weights(interface)
    pot = zeros(T, n_points, n_targets)
    for (i, pointi) in enumerate(eachpoint(interface))
        w_i = weights[i]
        for j in 1:n_targets
            target = (targets[1, j], targets[2, j])
            @inbounds pot[i, j] = laplace2d_pot(pointi.panel_point.point, target) * w_i
        end
    end
    return pot'
end

# fmm2d based fast evaluation
# matrix free via fmm2d, return a linear map
# potential: log(r), gradient: 1/r
function _laplace2d_S_fmm2d(charges::AbstractVector{Float64}, sources::Matrix{Float64}, weights::Vector{Float64}, thresh::Float64)
    n = length(charges)
    @assert size(sources) == (2, n)
    @assert size(weights) == (n,)
    vals = rfmm2d(eps = thresh, sources = sources, charges = weights .* charges, pg = 1)
    return vals.pot ./ 2π
end

function laplace2d_S_fmm2d(interface::DielectricInterface{P, Float64}, thresh::Float64) where {P <: AbstractPanel}
    n_points = num_points(interface)
    sources = zeros(Float64, 2, n_points)
    weights = zeros(Float64, n_points)
    for (i, point) in enumerate(eachpoint(interface))
        weights[i] = point.panel_point.weight
        sources[1, i] = point.panel_point.point[1]
        sources[2, i] = point.panel_point.point[2]
    end
    f = charges -> _laplace2d_S_fmm2d(charges, sources, weights, thresh)
    return LinearMap{Float64}(f, n_points, n_points)
end

function laplace2d_DT_fmm2d(interface::DielectricInterface{P, Float64}, thresh::Float64) where {P <: AbstractPanel}
    n_points = num_points(interface)
    sources = zeros(Float64, 2, n_points)
    weights = zeros(Float64, n_points)
    norms = zeros(Float64, 2, n_points)
    for (i, point) in enumerate(eachpoint(interface))
        weights[i] = point.panel_point.weight
        sources[1, i] = point.panel_point.point[1]
        sources[2, i] = point.panel_point.point[2]
        norms[1, i] = point.panel_point.normal[1]
        norms[2, i] = point.panel_point.normal[2]
    end

    gradn = zeros(Float64, n_points)
    function f(charges)
        vals = rfmm2d(eps = thresh, sources = sources, charges = weights .* charges, pg = 2)
        grad = vals.grad
        @inbounds for i in 1:n_points
            gradn[i] = norms[1, i] * grad[1, i] + norms[2, i] * grad[2, i]
        end
        return gradn ./ 2π
    end
    return LinearMap{Float64}(f, n_points, n_points)
end

function laplace2d_D_fmm2d(interface::DielectricInterface{P, Float64}, thresh::Float64) where {P <: AbstractPanel}
    n_points = num_points(interface)
    sources = zeros(Float64, 2, n_points)
    weights = zeros(Float64, n_points)
    norms = zeros(Float64, 2, n_points)
    for (i, point) in enumerate(eachpoint(interface))
        weights[i] = point.panel_point.weight
        sources[1, i] = point.panel_point.point[1]
        sources[2, i] = point.panel_point.point[2]
        norms[1, i] = point.panel_point.normal[1]
        norms[2, i] = point.panel_point.normal[2]
    end

    dipvecs = zeros(Float64, 2, n_points)
    function f(charges)
        @inbounds for i in 1:n_points
            cw = charges[i] * weights[i]
            dipvecs[1, i] = norms[1, i] * cw
            dipvecs[2, i] = norms[2, i] * cw
        end
        vals = rfmm2d(eps = thresh, sources = sources, dipvecs = dipvecs, pg = 1)
        return vals.pot ./ 2π
    end
    return LinearMap{Float64}(f, n_points, n_points)
end

function _laplace2d_pottrg_fmm2d(charges::AbstractVector{Float64}, sources::Matrix{Float64}, weights::Vector{Float64}, targets::Matrix{Float64}, thresh::Float64)
    n = length(charges)
    m = size(targets, 2)

    @assert size(sources) == (2, n)
    @assert size(targets) == (2, m)
    @assert size(weights) == (n,)
    vals = rfmm2d(eps = thresh, sources = sources, charges = weights .* charges, targets = targets, pgt = 1)
    return vals.pottarg ./ 2π
end

function laplace2d_pottrg_fmm2d(interface::DielectricInterface{P, Float64}, targets::Matrix{Float64}, thresh::Float64) where {P <: AbstractPanel}
    n_points = num_points(interface)
    sources = zeros(Float64, 2, n_points)
    weights = zeros(Float64, n_points)
    for (i, point) in enumerate(eachpoint(interface))
        weights[i] = point.panel_point.weight
        sources[1, i] = point.panel_point.point[1]
        sources[2, i] = point.panel_point.point[2]
    end

    f = charges -> _laplace2d_pottrg_fmm2d(charges, sources, weights, targets, thresh)
    return LinearMap{Float64}(f, size(targets, 2), n_points)
end
