# test/campaign/fixture_campaign.jl — a tiny self-contained campaign for pipeline tests.
# Writes a 6x6x6 Gaussian-blob template (3 cells/axis, 2 steps/cell), 2 orbitals one
# grid cell apart along x, into a tmp root.  Returns the path to the .toml file.

function write_fixture_xsf(path::AbstractString)
    open(path, "w") do io
        # CRYSTAL block: unit cell and one dummy atom
        println(io, "CRYSTAL")
        println(io, "PRIMVEC")
        println(io, "3.0 0.0 0.0")
        println(io, "0.0 3.0 0.0")
        println(io, "0.0 0.0 3.0")
        println(io, "PRIMCOORD")
        println(io, "1 1")
        println(io, "X 1.5 1.5 1.5")
        # Datagrid: 6x6x6 points, origin (0,0,0), spanning vectors 3.0 in each dir
        println(io, "BEGIN_BLOCK_DATAGRID_3D")
        println(io, " g")
        println(io, "BEGIN_DATAGRID_3D_g")
        println(io, "6 6 6")
        println(io, "0.0 0.0 0.0")
        println(io, "3.0 0.0 0.0")
        println(io, "0.0 3.0 0.0")
        println(io, "0.0 0.0 3.0")
        # Gaussian blob centred at (1.5,1.5,1.5); grid point (i-1)/6 * 3.0
        vals = Float64[]
        for k in 1:6, j in 1:6, i in 1:6
            x = ((i-1)/6)*3.0; y = ((j-1)/6)*3.0; z = ((k-1)/6)*3.0
            push!(vals, exp(-((x-1.5)^2 + (y-1.5)^2 + (z-1.5)^2) / (2*0.4^2)))
        end
        for chunk in Iterators.partition(vals, 6)
            println(io, join(string.(chunk), " "))
        end
        println(io, "END_DATAGRID_3D")
        println(io, "END_BLOCK_DATAGRID_3D")
    end
end

"""
    write_fixture_campaign(dir) -> toml_path

Write a self-contained tiny campaign into `dir`: one xsf template (Gaussian blob),
two orbitals one cell apart along x, neighbour cutoff 4.0 (keeps both on-site and
the (1,2) cross pair).  Returns the absolute path of the written `.toml`.
"""
function write_fixture_campaign(dir::AbstractString)
    dir = abspath(dir)
    xsf = joinpath(dir, "orb.xsf")
    write_fixture_xsf(xsf)
    out_root = joinpath(dir, "out")
    toml = joinpath(dir, "campaign.toml")
    # Orbitals: centroid is (1.8,1.8,1.8) (grid step 0.6, blob at grid index 3).
    # The true cell vector At = 3.0*(6/5) = 3.6, so orbital 2 is at centroid + At = (5.4,1.8,1.8).
    open(toml, "w") do io
        println(io, "name = \"mini\"")
        println(io, "root = \"$(out_root)\"")
        println(io, "templates = [\"$(xsf)\"]")
        println(io, "")
        println(io, "[[orbital]]")
        println(io, "type = 1")
        println(io, "x = 1.8")
        println(io, "y = 1.8")
        println(io, "z = 1.8")
        println(io, "")
        println(io, "[[orbital]]")
        println(io, "type = 1")
        println(io, "x = 5.4")
        println(io, "y = 1.8")
        println(io, "z = 1.8")
        println(io, "")
        println(io, "[pairing]")
        println(io, "neighbor_cutoff = 4.0")   # 3.6 < 4.0: keeps on-site + cross pair
        println(io, "")
        println(io, "[dielectrics]")
        println(io, "eps_out = 1.0")
        println(io, "boxes = [[1.5, 1.5, 1.5, 8.0, 8.0, 8.0, 2.0]]")
        println(io, "")
        println(io, "[solve]")
        println(io, "n_quad = 4")
        println(io, "edge_refine_level = 1")
        println(io, "rhs_tol = 1e-2")
        println(io, "lhs_tol = 1e-6")
        println(io, "gmres_rtol = 1e-8")
        println(io, "support_rtol = 1e-6")
        println(io, "volume_tol = 1e-8")
        println(io, "max_order = 8")
        println(io, "max_depth = 16")
        println(io, "")
        println(io, "[batching]")
        println(io, "n_centers_per_batch = 1")
        println(io, "")
        println(io, "[eval]")
        println(io, "far_pad_steps = 2.0")
    end
    return toml
end
