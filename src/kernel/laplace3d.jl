function laplace3d_pot(src::NTuple{3, T}, trg::NTuple{3, T}) where T
    r2 = sum((src .- trg).^2)
    r = sqrt(r2)
    inv_r = one(T) / r

    return inv_r / 4π
end

function laplace3d_grad(src::NTuple{3, T}, trg::NTuple{3, T}, norm::NTuple{3, T}) where T
    r2 = sum((src .- trg).^2)
    r = sqrt(r2)
    inv_r = one(T) / r

    return dot(norm, inv_r^3 .* (trg .- src))  / 4π
end

# direct evaluation of S, D and DT
function laplace3d_S(interface::DielectricInterface{P, T}) where {P <: AbstractPanel, T}
    n_points = num_points(interface)
    S = zeros(T, n_points, n_points)
    for (i, pointi) in enumerate(eachpoint(interface))
        for (j, pointj) in enumerate(eachpoint(interface))
            i == j && continue
            @inbounds S[i, j] = laplace3d_pot(pointj.panel_point.point, pointi.panel_point.point) * pointj.panel_point.weight
        end
    end
    return S
end

function laplace3d_DT(interface::DielectricInterface{P, T}) where {P <: AbstractPanel, T}
    n_points = num_points(interface)
    DT = zeros(T, n_points, n_points)
    for (i, pointi) in enumerate(eachpoint(interface))
        for (j, pointj) in enumerate(eachpoint(interface))
            i == j && continue
            @inbounds DT[i, j] = laplace3d_grad(pointj.panel_point.point, pointi.panel_point.point, pointi.panel_point.normal) * pointj.panel_point.weight
        end
    end
    return DT
end

function laplace3d_D(interface::DielectricInterface{P, T}) where {P <: AbstractPanel, T}
    n_points = num_points(interface)
    D = zeros(T, n_points, n_points)
    for (i, pointi) in enumerate(eachpoint(interface))
        for (j, pointj) in enumerate(eachpoint(interface))
            i == j && continue
            @inbounds D[j, i] = laplace3d_grad(pointj.panel_point.point, pointi.panel_point.point, pointi.panel_point.normal) * pointi.panel_point.weight
        end
    end
    return D
end

function laplace3d_D_trg(interface::DielectricInterface{P, T}, targets::Matrix{T}) where {P <: AbstractPanel, T}
    n_points = num_points(interface)
    n_targets = size(targets, 2)
    weights = all_weights(interface)
    D = zeros(T, n_points, n_targets)
    for (i, pointi) in enumerate(eachpoint(interface))
        w_i = weights[i]
        for j in 1:n_targets
            target = (targets[1, j], targets[2, j], targets[3, j])
            @inbounds D[i, j] = laplace3d_grad(pointi.panel_point.point, target, pointi.panel_point.normal) * w_i
        end
    end
    return D'
end

function laplace3d_pottrg(interface::DielectricInterface{P, T}, targets::Matrix{T}) where {P <: AbstractPanel, T}
    n_points = num_points(interface)
    n_targets = size(targets, 2)
    weights = all_weights(interface)
    pot = zeros(T, n_points, n_targets)
    for (i, pointi) in enumerate(eachpoint(interface))
        w_i = weights[i]
        for j in 1:n_targets
            target = (targets[1, j], targets[2, j], targets[3, j])
            @inbounds pot[i, j] = laplace3d_pot(pointi.panel_point.point, target) * w_i
        end
    end
    return pot'
end

function laplace3d_DT_fmm3d(interface::DielectricInterface{P, Float64}, thresh::Float64) where {P <: AbstractPanel}
    n_points = num_points(interface)
    sources = zeros(Float64, 3, n_points)
    weights = zeros(Float64, n_points)
    norms = zeros(Float64, 3, n_points)
    for (i, point) in enumerate(eachpoint(interface))
        weights[i] = point.panel_point.weight
        sources[1, i] = point.panel_point.point[1]
        sources[2, i] = point.panel_point.point[2]
        sources[3, i] = point.panel_point.point[3]
        norms[1, i] = point.panel_point.normal[1]
        norms[2, i] = point.panel_point.normal[2]
        norms[3, i] = point.panel_point.normal[3]
    end

    gradn = zeros(Float64, n_points)
    function f(charges)
        vals = lfmm3d(thresh, sources, charges = weights .* charges, pg = 2)
        grad = vals.grad
        @inbounds for i in 1:n_points
            gradn[i] = norms[1, i] * grad[1, i] + norms[2, i] * grad[2, i] + norms[3, i] * grad[3, i]
        end
        return -gradn ./ 4π
    end
    return LinearMap{Float64}(f, n_points, n_points)
end

function laplace3d_D_fmm3d(interface::DielectricInterface{P, Float64}, thresh::Float64) where {P <: AbstractPanel}
    n_points = num_points(interface)
    sources = zeros(Float64, 3, n_points)
    weights = zeros(Float64, n_points)
    norms = zeros(Float64, 3, n_points)
    for (i, point) in enumerate(eachpoint(interface))
        weights[i] = point.panel_point.weight
        sources[1, i] = point.panel_point.point[1]
        sources[2, i] = point.panel_point.point[2]
        sources[3, i] = point.panel_point.point[3]
        norms[1, i] = point.panel_point.normal[1]
        norms[2, i] = point.panel_point.normal[2]
        norms[3, i] = point.panel_point.normal[3]
    end

    dipvecs = zeros(Float64, 3, n_points)
    function f(charges)
        @inbounds for i in 1:n_points
            cw = charges[i] * weights[i]
            dipvecs[1, i] = norms[1, i] * cw
            dipvecs[2, i] = norms[2, i] * cw
            dipvecs[3, i] = norms[3, i] * cw
        end
        vals = lfmm3d(thresh, sources, dipvecs = dipvecs, pg = 1)
        return -vals.pot ./ 4π
    end
    return LinearMap{Float64}(f, n_points, n_points)
end

function laplace3d_D_trg_fmm3d(interface::DielectricInterface{P, Float64}, targets::Matrix{Float64}, thresh::Float64) where {P <: AbstractPanel}
    n_points = num_points(interface)
    sources = zeros(Float64, 3, n_points)
    weights = zeros(Float64, n_points)
    norms = zeros(Float64, 3, n_points)
    for (i, point) in enumerate(eachpoint(interface))
        weights[i] = point.panel_point.weight
        sources[1, i] = point.panel_point.point[1]
        sources[2, i] = point.panel_point.point[2]
        sources[3, i] = point.panel_point.point[3]
        norms[1, i] = point.panel_point.normal[1]
        norms[2, i] = point.panel_point.normal[2]
        norms[3, i] = point.panel_point.normal[3]
    end

    dipvecs = zeros(Float64, 3, n_points)
    function f(charges)
        @inbounds for i in 1:n_points
            cw = charges[i] * weights[i]
            dipvecs[1, i] = norms[1, i] * cw
            dipvecs[2, i] = norms[2, i] * cw
            dipvecs[3, i] = norms[3, i] * cw
        end
        vals = lfmm3d(thresh, sources, dipvecs = dipvecs, targets = targets, pgt = 1)
        return vals.pottarg ./ 4π
    end
    return LinearMap{Float64}(f, size(targets, 2), n_points)
end

function _laplace3d_pottrg_fmm3d(charges::AbstractVector{Float64}, sources::Matrix{Float64}, weights::Vector{Float64}, targets::Matrix{Float64}, thresh::Float64)
    n = length(charges)
    @assert size(sources) == (3, n)
    @assert size(targets, 1) == 3
    @assert size(weights) == (n,)
    vals = lfmm3d(thresh, sources, charges = weights .* charges, targets = targets, pgt = 1)
    return vals.pottarg ./ 4π
end

function laplace3d_pottrg_fmm3d(interface::DielectricInterface{P, Float64}, targets::Matrix{Float64}, thresh::Float64) where {P <: AbstractPanel}
    n_points = num_points(interface)
    sources = zeros(Float64, 3, n_points)
    weights = zeros(Float64, n_points)
    for (i, point) in enumerate(eachpoint(interface))
        weights[i] = point.panel_point.weight
        sources[1, i] = point.panel_point.point[1]
        sources[2, i] = point.panel_point.point[2]
        sources[3, i] = point.panel_point.point[3]
    end

    f = charges -> _laplace3d_pottrg_fmm3d(charges, sources, weights, targets, thresh)
    return LinearMap{Float64}(f, size(targets, 2), n_points)
end
