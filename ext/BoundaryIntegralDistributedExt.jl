module BoundaryIntegralDistributedExt

using BoundaryIntegral
using Distributed
using SlurmClusterManager

# Spawn workers: SlurmManager() inside a Slurm allocation (one worker/task, reads env),
# local addprocs(workers) otherwise, or none (inline) when workers==0 and not on Slurm.
function _spawn(workers::Int)
    proj = dirname(Base.active_project())
    glue = get(ENV, "JULIA_GLUE_THREADS", "8")
    if haskey(ENV, "SLURM_JOB_ID") && parse(Int, get(ENV, "SLURM_NTASKS", "1")) > 1
        @info "run_phase: SlurmManager workers"
        addprocs(SlurmManager(); exeflags = `--project=$proj -t $glue`)
    elseif workers > 0
        @info "run_phase: $workers local workers"
        addprocs(workers; exeflags = `--project=$proj -t $glue`)
    end
end

function BoundaryIntegral.run_phase(c::BoundaryIntegral.CampaignInput, phase::Symbol; workers::Int = 0)
    phase in (:solve, :eval) || error("run_phase: phase must be :solve or :eval")
    _spawn(workers)
    @everywhere eval(:(using BoundaryIntegral))
    # workers buffer non-TTY stdio (Julia 1.12) — flush periodically so their progress
    # output reaches the master/log live ( @everywhere ships an AST, not a closure ✓ )
    @everywhere Timer(_ -> (flush(stdout); flush(stderr)), 2; interval = 2)
    pending = pending_batches(c, phase)
    @info "run_phase: dispatching" phase npending=length(pending) nworkers=nworkers()
    flush(stderr); flush(stdout)
    # Only core-package objects may cross the wire: workers never load this extension, so
    # an ext-owned closure deserialization-fails there and wedges pmap (see PhaseRunner).
    # That includes on_error — pmap wraps it INTO the function it ships (wrap_on_error
    # runs before remote() in pmap), so it must be a named non-ext function: identity.
    results = pmap(BoundaryIntegral.PhaseRunner(c.toml_path, phase), pending;
                   retry_delays = [30.0], on_error = identity)
    ok = count(r -> r isa Tuple && r[2] === :ok, results)
    @info "run_phase finished" phase ok failed=length(results)-ok
    flush(stderr); flush(stdout)
    return results
end

end
