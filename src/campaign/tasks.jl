# src/campaign/tasks.jl
# Campaign phase functions: prepare, solve_batch, consolidate, eval_batch, assemble_v,
# pending_batches — plus *_core variants for in-memory reuse and four_index_integrals.
# Adapted from CampaignLib/tasks.jl; uses CampaignInput from toml_input.jl.

using Sockets

# ---------------------------------------------------------------------------
# Params fingerprint (no nx/ny; hashes explicit orbital list)
# ---------------------------------------------------------------------------

_params_string(c::CampaignInput) =
    "norb=$(length(c.orbitals)) cutoff=$(c.neighbor_cutoff) n_per_batch=$(c.n_centers_per_batch) overrides=$(c.pair_overrides)"

# Manifest derivation shared by `prepare` (file-based) and `four_index_integrals`
# (in-memory): centers, neighbor pairs (or explicit overrides), and batches.
function _centers_pairs_batches(c::CampaignInput)
    centers = enumerate_centers(c)
    pairs   = c.pair_overrides === nothing ?
        enumerate_pairs(centers, c.neighbor_cutoff) : c.pair_overrides
    batches = build_batches(pairs, c.n_centers_per_batch)
    return centers, pairs, batches
end

# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

"""
    prepare(c::CampaignInput)

Phase 1 (single process): enumerate centers/pairs/batches and write
`centers.tsv` + `manifest.tsv` under `c.root`. Idempotent: existing files are kept
(delete them to re-prepare). Guards against parameter drift via `manifest.params`.
"""
function prepare(c::CampaignInput)
    params_file = joinpath(c.root, "manifest.params")
    if isfile(manifest_path(c)) && isfile(centers_path(c))
        @info "prepare: manifest exists, skipping" manifest_path(c)
        if isfile(params_file)
            stored = strip(read(params_file, String))
            current = _params_string(c)
            stored == current || error(
                "Campaign parameters changed since last prepare.\n" *
                "  stored : $stored\n" *
                "  current: $current\n" *
                "Delete the manifest files in $(c.root) to re-prepare.")
        else
            @warn "prepare: manifest.params missing (old campaign); skipping parameter check"
        end
        return read_manifest(manifest_path(c))
    end
    centers, pairs, batches = _centers_pairs_batches(c)
    mkpath(c.root)
    write_centers(centers_path(c), centers)
    write_manifest(manifest_path(c), batches)
    write(params_file, _params_string(c))
    @info "prepare: wrote manifest" n_centers=length(centers) n_pairs=length(pairs) n_batches=length(batches)
    return batches
end

# ---------------------------------------------------------------------------
# pending_batches
# ---------------------------------------------------------------------------

"""
    pending_batches(c::CampaignInput, phase::Symbol) -> Vector{Int}

Batch ids still to do, derived from files on disk.
`:solve` → no complete batch file; `:eval` → no complete V file.
"""
function pending_batches(c::CampaignInput, phase::Symbol)
    isfile(manifest_path(c)) || error("manifest not found at $(manifest_path(c)); run prepare(c) first")
    batches = read_manifest(manifest_path(c))
    ids = getfield.(batches, :batch_id)
    if phase === :solve
        return [id for id in ids if !is_complete_batch(batch_path(c, id))]
    elseif phase === :eval
        return [id for id in ids if !_is_complete_v(v_path(c, id))]
    end
    error("unknown phase $phase")
end

# ---------------------------------------------------------------------------
# V file helpers
# ---------------------------------------------------------------------------

const V_FORMAT_VERSION = 1

function save_v_rows(path::AbstractString, batch_id::Int,
        source_pairs::Vector{Tuple{Int,Int}}, target_pairs::Vector{Tuple{Int,Int}},
        V::Matrix{Float64}, stats::Dict{String,Any})
    _atomic_serialize(path, (; version = V_FORMAT_VERSION, batch_id,
        source_pairs, target_pairs, V, stats))
end

function load_v_rows(path::AbstractString)
    vr = open(deserialize, path)
    vr.version == V_FORMAT_VERSION || error("$path: V format version mismatch")
    size(vr.V) == (length(vr.target_pairs), length(vr.source_pairs)) ||
        error("$path: V shape mismatch")
    return vr
end

function _is_complete_v(path::AbstractString)
    isfile(path) || return false
    try
        load_v_rows(path)
        return true
    catch
        return false
    end
end

# ---------------------------------------------------------------------------
# _atomic_serialize
# ---------------------------------------------------------------------------

function _atomic_serialize(path::AbstractString, obj)
    d = dirname(path)
    isempty(d) || mkpath(d)
    tmp = string(path, ".tmp.", getpid(), "_", rand(UInt32))
    open(io -> serialize(io, obj), tmp, "w")
    mv(tmp, path; force = true)
end

# Atomic text write (tmp + rename); `writer` is called with the open IO. `writer` is the
# first argument (like `open(f, path)`) so callers can use do-block syntax. Text analogue
# of `_atomic_serialize` — shared by the V table and report so every atomic write uses
# the same collision-safe pid+rand tmp name.
function _atomic_write_text(writer, path::AbstractString)
    d = dirname(path)
    isempty(d) || mkpath(d)
    tmp = string(path, ".tmp.", getpid(), "_", rand(UInt32))
    open(writer, tmp, "w")
    mv(tmp, path; force = true)
    return path
end

# Cartesian coordinates (3 x length(gidx)) of integer grid points `gidx` on datagrid `dg`.
function grid_positions(dg, gidx)
    positions = Matrix{Float64}(undef, 3, length(gidx))
    for (r, g) in enumerate(gidx)
        p = grid_point(dg, g[1], g[2], g[3])
        positions[1, r] = p[1]; positions[2, r] = p[2]; positions[3, r] = p[3]
    end
    return positions
end

# ---------------------------------------------------------------------------
# _batch_instances helper
# ---------------------------------------------------------------------------

# OrbitalInstances needed by a batch: the unique orbital ids across the batch's pairs,
# each from its center record. `byid` may come from disk (read_centers) or in-memory
# (enumerate_centers) — both yield center records with .template_id and .steps.
function _insts_for(byid::AbstractDict, spec::BatchSpec)
    need = unique(reduce(vcat, [[p[1], p[2]] for p in spec.pairs]; init = Int[]))
    return Dict(id => OrbitalInstance(id, byid[id].template_id, byid[id].steps) for id in need)
end

function _batch_instances(c::CampaignInput, spec::BatchSpec)
    centers = read_centers(centers_path(c))
    byid = Dict(ct.id => ct for ct in centers)
    return _insts_for(byid, spec)
end

# ---------------------------------------------------------------------------
# solve_batch_core  (in-memory: assemble + solve + pack → BatchResult, no file)
# ---------------------------------------------------------------------------

"""
    solve_batch_core(c, spec, grids, insts) -> BatchResult

Assemble pair densities, build the shared envelope interface, block-GMRES, and pack
the result. No file IO — the caller decides whether to persist.
"""
function solve_batch_core(c::CampaignInput, spec::BatchSpec,
        grids::AbstractVector, insts::AbstractDict{Int,OrbitalInstance})
    b = assemble_lattice_batch(grids, insts, spec.pairs;
        support_rtol = c.solve["support_rtol"])
    res = solve_dielectric_lattice_batch(c.boxes, c.epses, c.eps_out, b;
        n_quad = Int(c.solve["n_quad"]), rhs_atol = c.solve["rhs_tol"],
        l_ec = campaign_l_ec(c), fmm_tol = c.solve["lhs_tol"],
        up_tol = c.solve["lhs_tol"], max_order = Int(c.solve["max_order"]),
        gmres_rtol = c.solve["gmres_rtol"], max_depth = Int(c.solve["max_depth"]))
    stats = Dict{String,Any}(
        "niter" => res.stats.niter, "dof" => size(res.sigma, 1),
        "n_support" => length(b.gidx), "K" => length(spec.pairs),
        "hostname" => gethostname())
    return BatchResult(BATCH_FORMAT_VERSION, spec.batch_id, b.pair_ids,
        b.gidx, b.weights, b.densities, res.interface, res.sigma, stats)
end

# ---------------------------------------------------------------------------
# solve_batch  (file-based wrapper around solve_batch_core)
# ---------------------------------------------------------------------------

"""
    solve_batch(c::CampaignInput, batch_id) -> path | nothing

Solve phase for one batch: assemble pair densities, envelope-refine ONE shared
interface, block-GMRES, write the BatchResult atomically. Skips if output already
exists.
"""
function solve_batch(c::CampaignInput, batch_id::Int)
    out = batch_path(c, batch_id)
    if is_complete_batch(out)
        @info "solve_batch: already complete, skipping" batch_id
        return nothing
    end
    t0 = time()
    found = filter(b -> b.batch_id == batch_id, read_manifest(manifest_path(c)))
    isempty(found) && error("batch_id=$batch_id not found in manifest $(manifest_path(c))")
    spec = only(found)
    temps = load_templates!(c)
    grids = [t[2] for t in temps]
    insts = _batch_instances(c, spec)
    t_setup = time() - t0

    br = solve_batch_core(c, spec, grids, insts)
    # patch timing into stats
    br.stats["t_setup"] = t_setup
    br.stats["t_total"] = time() - t0

    save_batch_result(out, br)
    @info "solve_batch: done" batch_id dof=br.stats["dof"] K=br.stats["K"] t_total=br.stats["t_total"]
    return out
end

# ---------------------------------------------------------------------------
# consolidate_core  (in-memory: build targets + rho_store from a vector of BatchResults)
# ---------------------------------------------------------------------------

"""
    consolidate_core(brs, dg, c) -> (targets, store)

Build the shared eval target set and the contraction store from a vector of
BatchResults. `targets = (; gidx, positions)`, `store = (; pair_ids, t_idx, tw)`.
No file IO.
"""
function consolidate_core(brs::Vector{BatchResult}, dg, c::CampaignInput)
    gset = Set{NTuple{3,Int}}()
    for br in brs
        union!(gset, br.gidx)
    end
    gidx = sort!(collect(gset))
    rowofg = Dict(g => r for (r, g) in enumerate(gidx))

    targets = (; gidx, positions = grid_positions(dg, gidx))

    pair_ids = Tuple{Int,Int}[]
    t_idx    = Vector{Vector{Int}}()
    tw       = Vector{Vector{Float64}}()
    for br in brs
        rows = [rowofg[g] for g in br.gidx]
        for k in 1:length(br.pair_ids)
            push!(pair_ids, br.pair_ids[k])
            push!(t_idx, copy(rows))
            push!(tw, br.weights .* br.densities[:, k])
        end
    end
    store = (; pair_ids, t_idx, tw)
    return targets, store
end

# ---------------------------------------------------------------------------
# consolidate  (file-based wrapper around consolidate_core)
# ---------------------------------------------------------------------------

"""
    consolidate(c::CampaignInput)

Between solve and eval (single process): build
- `targets.jls`: `(; gidx, positions)` — the sorted union of ALL stored batch supports;
- `rho_store.jls`: `(; pair_ids, t_idx, tw)` — per pair, its support's row indices in T
  and the contraction vector `w .* ρ`, in manifest batch order.
Errors if any batch is missing. Idempotent.
"""
function consolidate(c::CampaignInput)
    batches = read_manifest(manifest_path(c))
    missing_ids = [b.batch_id for b in batches if !is_complete_batch(batch_path(c, b.batch_id))]
    isempty(missing_ids) || error("consolidate: unsolved batches: $missing_ids")

    brs  = [load_batch_result(batch_path(c, b.batch_id)) for b in batches]
    temps = load_templates!(c)
    dg   = temps[1][2]

    targets, store = consolidate_core(brs, dg, c)
    _atomic_serialize(targets_path(c), targets)
    _atomic_serialize(rho_store_path(c), store)
    @info "consolidate: done" n_targets=length(targets.gidx) n_pairs=length(store.pair_ids)
    return nothing
end

# ---------------------------------------------------------------------------
# eval_batch_core  (in-memory: Φ-eval + contraction → (source_pairs, V), no file)
# ---------------------------------------------------------------------------

"""
    eval_batch_core(br, targets, store, dg, c) -> (source_pairs, V)

Evaluate `Φ_a = u_inc[ρ_a] + u[σ_a]` at the shared target set `T`, then contract
against every stored pair density. Returns `(br.pair_ids, nP × K Matrix)`. No file IO.
"""
function eval_batch_core(br::BatchResult, targets, store, dg, c::CampaignInput)
    K = length(br.pair_ids)
    pos = grid_positions(dg, br.gidx)
    sources = [VolumeSource(copy(pos), copy(br.weights), br.densities[:, k]) for k in 1:K]

    At, Bt, Ct = true_cell_vectors(dg)
    max_step = maximum((norm(collect(At)) / dg.nx,
                        norm(collect(Bt)) / dg.ny,
                        norm(collect(Ct)) / dg.nz))
    far_pad = c.far_pad_steps * max_step

    Φ = evaluate_batch_potential(br.interface, br.sigma, sources, targets.positions;
        lhs_tol   = c.solve["lhs_tol"],
        volume_tol = c.solve["volume_tol"],
        far_pad    = far_pad,
        screen_boxes = c.boxes, screen_epses = c.epses, screen_eps_out = c.eps_out)

    nP = length(store.pair_ids)
    V  = Matrix{Float64}(undef, nP, K)
    for kl in 1:nP, a in 1:K
        V[kl, a] = dot(store.tw[kl], view(Φ, store.t_idx[kl], a))
    end
    return br.pair_ids, V
end

# ---------------------------------------------------------------------------
# eval_batch  (file-based wrapper around eval_batch_core)
# ---------------------------------------------------------------------------

"""
    eval_batch(c::CampaignInput, batch_id) -> path | nothing

Post-eval phase for one batch: rebuild K sources from the BatchResult, evaluate
Φ_a at the shared target set T, contract against every stored pair density, write
K columns of V atomically. Skips if V file already complete.
"""
function eval_batch(c::CampaignInput, batch_id::Int)
    out = v_path(c, batch_id)
    if _is_complete_v(out)
        @info "eval_batch: already complete, skipping" batch_id
        return nothing
    end
    t0 = time()
    br = load_batch_result(batch_path(c, batch_id))

    isfile(targets_path(c)) && isfile(rho_store_path(c)) ||
        error("eval_batch: targets.jls / rho_store.jls not found under $(c.root); run consolidate(c) first")

    targets = open(deserialize, targets_path(c))
    store   = open(deserialize, rho_store_path(c))
    temps   = load_templates!(c)
    dg      = temps[1][2]

    t_setup = time() - t0
    t_phi_start = time()
    source_pairs, V = eval_batch_core(br, targets, store, dg, c)
    t_phi = time() - t_phi_start

    stats = Dict{String,Any}(
        "t_setup" => t_setup, "t_phi" => t_phi, "t_total" => time() - t0,
        "n_targets" => size(targets.positions, 2), "hostname" => gethostname())
    save_v_rows(out, batch_id, source_pairs, store.pair_ids, V, stats)
    @info "eval_batch: done" batch_id n_targets=size(targets.positions, 2) t_total=stats["t_total"]
    return out
end

# ---------------------------------------------------------------------------
# assemble_v
# ---------------------------------------------------------------------------

"""
    assemble_v(c::CampaignInput) -> (; max_rel_asym, n)

Final phase (single process): gather all V files into the dense `n_pairs × n_pairs`
matrix, write `V_full.tsv` and `report.txt` with the max-rel-asymmetry diagnostic.
"""
function assemble_v(c::CampaignInput)
    batches = read_manifest(manifest_path(c))
    missing_ids = [b.batch_id for b in batches if !_is_complete_v(v_path(c, b.batch_id))]
    isempty(missing_ids) || error("assemble_v: unevaluated batches: $missing_ids — run eval_batch for each first")

    store    = open(deserialize, rho_store_path(c))
    pair_ids = store.pair_ids
    col      = Dict(p => i for (i, p) in enumerate(pair_ids))
    n        = length(pair_ids)
    V        = fill(NaN, n, n)
    for b in batches
        vr = load_v_rows(v_path(c, b.batch_id))
        vr.target_pairs == pair_ids || error("V_$(b.batch_id): target ordering mismatch")
        for (k, sp) in enumerate(vr.source_pairs)
            V[:, col[sp]] = vr.V[:, k]
        end
    end
    any(isnan, V) && error("assemble_v: missing columns (run eval for all batches first)")

    scale        = maximum(abs.(V))
    max_rel_asym = maximum(abs.(V .- transpose(V))) / scale
    write_v_table(joinpath(c.root, "V_full.tsv"), pair_ids, V)

    # eV-unit tensor (Hubbard-like normalization), matching the ScreenedOrbitalSolve.to_eV
    # convention: V_eV[a,b] = V_raw[a,b] * 4π * E2 / (Na * Nb), with E2 = e^2/(4πε0) =
    # 14.3996 eV·Å and Na = ∫ρ_a = sum(store.tw[a]) the pair-density norm. Stored as the
    # primary tensor in V_full_eV.jls; V_full.tsv keeps the raw (1/4π-kernel) values.
    E2 = 14.3996
    Na = [sum(store.tw[a]) for a in 1:n]
    V_eV = Matrix{Float64}(undef, n, n)
    for a in 1:n, bb in 1:n
        V_eV[a, bb] = V[a, bb] * 4π * E2 / (Na[a] * Na[bb])
    end
    _atomic_serialize(joinpath(c.root, "V_full_eV.jls"),
        (; pair_ids = pair_ids, V = V_eV, norms = Na, unit = "eV", E2_eV_Ang = E2))
    write_v_table(joinpath(c.root, "V_full_eV.tsv"), pair_ids, V_eV)

    report = sprint() do io
        println(io, "campaign: $(c.name)")
        println(io, "pairs: $n   batches: $(length(batches))")
        println(io, "max|V| (raw): $scale")
        println(io, "max rel asymmetry |V - V'|/max|V|: $max_rel_asym")
        println(io, "max|V| (eV):  $(maximum(abs.(V_eV)))")
        println(io, "onsite V[1,1] (eV): $(V_eV[1,1])")
    end
    _atomic_write_text(io -> print(io, report), joinpath(c.root, "report.txt"))
    @info "assemble_v: done" n max_rel_asym
    return (; max_rel_asym, n, V_eV, pair_ids)
end

# ---------------------------------------------------------------------------
# four_index_integrals  (fully in-memory — no files written)
# ---------------------------------------------------------------------------

"""
    four_index_integrals(c::CampaignInput) -> (; pair_ids, V)

Run the full four-index pipeline in memory (no files written).  Reuses the same
`*_core` functions as the file-based phases to guarantee identical numerics.
"""
function four_index_integrals(c::CampaignInput)
    centers, _, batches = _centers_pairs_batches(c)
    byid    = Dict(ct.id => ct for ct in centers)
    temps   = load_templates!(c)
    grids   = [t[2] for t in temps]
    dg      = grids[1]

    brs = [solve_batch_core(c, spec, grids, _insts_for(byid, spec)) for spec in batches]
    targets, store = consolidate_core(brs, dg, c)

    pid = store.pair_ids
    col = Dict(p => i for (i, p) in enumerate(pid))
    V   = fill(NaN, length(pid), length(pid))
    for br in brs
        sp_pairs, Vb = eval_batch_core(br, targets, store, dg, c)
        for (k, sp) in enumerate(sp_pairs)
            V[:, col[sp]] = Vb[:, k]
        end
    end
    return (; pair_ids = pid, V)
end

"""
    four_index_integrals(toml_path::AbstractString) -> (; pair_ids, V)

Load a campaign from `toml_path` and run the full four-index pipeline in memory.
"""
four_index_integrals(toml_path::AbstractString) =
    four_index_integrals(load_campaign(toml_path))

# ---------------------------------------------------------------------------
# run_phase stub (implemented by BoundaryIntegralDistributedExt)
# ---------------------------------------------------------------------------

"""
    run_phase(c, phase; workers=0)

Parallel driver for `:solve`/`:eval` over `pending_batches`. Implemented by the
`BoundaryIntegralDistributedExt` extension (load `Distributed` + `SlurmClusterManager`).
Without the extension loaded, this errors with a hint.
"""
function run_phase end

"""
    PhaseRunner(toml_path, phase)

Callable mapped over pending batch ids by `run_phase`. Lives in the core package —
not the Distributed extension — because `pmap` serializes its function to workers,
and workers load only `BoundaryIntegral` (never the extension, which needs
SlurmClusterManager). An extension-owned closure cannot be deserialized there: the
worker hits `KeyError: PkgId(... BoundaryIntegralDistributedExt) not found`, the
failure wedges the message stream, and `pmap` waits forever.
"""
struct PhaseRunner
    toml::String
    phase::Symbol
end

function (r::PhaseRunner)(id::Integer)
    runner = r.phase === :solve ? solve_batch :
             r.phase === :eval  ? eval_batch  :
             error("PhaseRunner: phase must be :solve or :eval")
    try
        runner(load_campaign(r.toml), id)
        return (id, :ok)
    catch err
        c = load_campaign(r.toml)
        mkpath(logs_dir(c))
        write(joinpath(logs_dir(c), "$(r.phase)_batch_$(lpad(id, 4, '0')).err"),
              sprint(showerror, err, catch_backtrace()))
        rethrow()
    end
end
