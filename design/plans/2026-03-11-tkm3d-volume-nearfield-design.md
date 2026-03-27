# TKM3D Volume Near-Field Design

**Date:** 2026-03-11

**Goal:** Remove `FBCPoisson` from the volume-source box3d solver path and use `TKM3D.ltkm3dc` to evaluate nearby surface targets instead.

## Scope

- Remove `using FBCPoisson` from the package entrypoint.
- Replace all `lfbc3d` usage in the box3d volume-source hybrid evaluator with `ltkm3dc`.
- Preserve the existing near/far target classification and adaptive panel refinement logic.
- Keep `FMM3D.lfmm3d` as the far-field backend.
- Update tests and planning docs to reflect the backend change.

## Chosen Approach

The current near/far split in `src/shape/box3d.jl` remains in place. Far targets continue to use `lfmm3d(..., pgt = 2)`. Near targets are evaluated with `ltkm3dc(..., pgt = 2)`, using preweighted source charges from the existing `VolumeSource` helpers.

To keep the TKM evaluation stable for irregular source clouds, the integration adds a small helper to derive an explicit `kmax` from the estimated source spacing. That helper uses the same spacing estimate already used by near/far classification, with a fallback based on quadrature weights for sparse or degenerate source sets.

## Numerical Handling

- `lfmm3d` gradients are interpreted with the existing `1 / (4pi)` normalization.
- `ltkm3dc` returns free-space Laplace target gradients, so the implementation uses the same `1 / (4pi)` scaling in the RHS assembly path.
- The `rhs` sign convention remains unchanged.

## Files

- Modify `src/BoundaryIntegral.jl`
- Modify `src/shape/box3d.jl`
- Modify `test/solver/dielectric_box3d.jl`
- Keep the existing `Project.toml` dependency swap to `TKM3D`

## Risks

- `ltkm3dc` accuracy depends on `kmax`; an explicit helper is needed instead of relying on the package default.
- The repo already contains an uncommitted `Project.toml` dependency edit, so implementation should preserve that change rather than overwrite it.

## Verification

- Add a regression test that asserts the codebase is wired to `TKM3D/ltkm3dc` and no longer references `FBCPoisson/lfbc3d`.
- Run targeted box3d solver tests after the backend swap.
