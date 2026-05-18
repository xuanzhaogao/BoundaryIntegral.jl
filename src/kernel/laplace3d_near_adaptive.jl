# Adaptive quadtree-based moments for touching (edge-sharing) near pairs.
# See docs/superpowers/specs/2026-05-18-near-correction-improvements-design.md §5.4.

struct AdaptiveConfig
    atol::Float64
    rtol::Float64
    n_GL::Int          # base GL order per leaf; 0 means "use source panel's n_quad"
    max_depth::Int
end

AdaptiveConfig(; atol::Float64, rtol::Float64 = sqrt(eps(Float64)),
                 n_GL::Int = 0, max_depth::Int = 20) =
    AdaptiveConfig(atol, rtol, n_GL, max_depth)
