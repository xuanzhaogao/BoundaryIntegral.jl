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
    runner = phase === :solve ? BoundaryIntegral.solve_batch :
             phase === :eval  ? BoundaryIntegral.eval_batch  :
             error("run_phase: phase must be :solve or :eval")
    _spawn(workers)
    @everywhere eval(:(using BoundaryIntegral))
    pending = pending_batches(c, phase)
    toml = c.toml_path                         # workers reload from the .toml path (cheap, cached)
    results = pmap(pending; retry_delays = [30.0], on_error = e -> e) do id
        try
            runner(BoundaryIntegral.load_campaign(toml), id)
            (id, :ok)
        catch err
            cc = BoundaryIntegral.load_campaign(toml)
            mkpath(logs_dir(cc))
            write(joinpath(logs_dir(cc), "$(phase)_batch_$(lpad(id, 4, '0')).err"),
                  sprint(showerror, err, catch_backtrace()))
            rethrow()
        end
    end
    ok = count(r -> r isa Tuple && r[2] === :ok, results)
    @info "run_phase finished" phase ok failed=length(results)-ok
    return results
end

end
