# Pointwise RHS Refinement Near/Far Classification Design

**Problem:** `single_dielectric_box3d_rhs_adaptive` classifies whole temporary panels as near or far during RHS-based refinement. For large faces, a single near panel center causes every quadrature and verification target on that panel to be sent through TKM, which expands the NUFFT bounding box to the full face and can trigger extreme memory use.

**Goal:** Classify generated quadrature and verification targets individually during RHS refinement so TKM only receives the truly near targets, matching the existing runtime hybrid evaluator behavior.

## Context

- Runtime hybrid evaluation in `src/solver/dielectric_box3d.jl` already classifies targets pointwise with `_classify_near_far_targets`.
- RHS refinement in `src/shape/box3d.jl` still calls `_classify_near_far_panels` and then promotes one Boolean per panel to all generated targets on that panel.
- The orbital reproduction shows that two large top/bottom panels can become "near", producing a `90 x 90` target box for TKM even though only a small subset of points is actually near the source cloud.

## Approaches Considered

### 1. Pointwise target classification in refinement

Generate all quadrature and verification targets first, classify them with `_classify_near_far_targets`, and pass the resulting mask directly into `_rhs_volume_targets_hybrid`.

Pros:
- Minimal behavioral change.
- Reuses existing target-level helper and runtime logic.
- Fixes the oversized TKM box without changing the near-radius rule.

Cons:
- Adds one KD-tree query per generated target during refinement.

### 2. Force early panel subdivision before near/far classification

Split panels geometrically first, then keep panel-level classification on smaller patches.

Pros:
- Preserves coarse panel abstraction.

Cons:
- More invasive change to the refinement loop.
- Still approximate at the panel level.
- Harder to reason about interaction with current convergence checks.

### 3. Change the near-radius heuristic

Keep panel-level classification but shrink or otherwise alter the `h_factor * h` rule.

Pros:
- Small code delta.

Cons:
- Does not address the root issue that one near panel center promotes all panel targets.
- Risks accuracy regressions by changing the physical split criterion.

## Recommended Design

Adopt approach 1. Keep the existing `distance <= h_factor * h` criterion, but apply it to the generated targets instead of to temporary panels in RHS refinement. Runtime hybrid evaluation already does this, so the change makes refinement consistent with the evaluation path rather than introducing a new heuristic.

## Testing

- Add a regression test showing that pointwise classification on a large face marks only a subset of generated targets as near.
- Add a refinement-oriented test that compares panel-level classification to pointwise classification and proves they differ for a localized source near a large face.
- Keep the existing runtime hybrid tests to confirm pointwise classification behavior remains intact there.
