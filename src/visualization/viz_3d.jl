function viz_3d(args...)
    @warn "implemented as extension, please load a Makie backend such as `using CairoMakie` or `using GLMakie` to use this function"
    return nothing
end

function viz_3d_surface(args...)
    @warn "implemented as extension, please load a Makie backend such as `using CairoMakie` or `using GLMakie` to use this function"
    return nothing
end

function viz_3d_interface_solution(args...)
    @warn "implemented as extension, please load a Makie backend such as `using CairoMakie` or `using GLMakie` to use this function"
    return nothing
end

function viz_3d_zslice(args...)
    @warn "implemented as extension, please load a Makie backend such as `using CairoMakie` or `using GLMakie` to use this function"
    return nothing
end

"""
    plot_campaign_geometry(c::CampaignInput; kwargs...) -> Figure
    plot_campaign_geometry(toml_path::AbstractString; kwargs...) -> Figure

3D view of a campaign's geometry: each dielectric box as a wireframe cuboid colored by its
permittivity ε, plus the orbital positions as a scatter colored by sublattice `type`. Requires
a Makie backend (`using CairoMakie`/`GLMakie`). Implemented in the Makie extension.
"""
function plot_campaign_geometry(args...; kwargs...)
    @warn "implemented as extension, please load a Makie backend such as `using CairoMakie` or `using GLMakie` to use this function"
    return nothing
end
