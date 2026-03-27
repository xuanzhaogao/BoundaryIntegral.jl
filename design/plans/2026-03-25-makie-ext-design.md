# Makie Extension Refactor Design

**Problem:** Visualization support is currently wired through a `GLMakie`-specific package extension and user messaging, which fixes the rendering backend to `GLMakie` even though the plotting code itself should be backend-agnostic.

**Goal:** Replace the current `GLMakie` extension with a `Makie` extension so `BoundaryIntegral` depends only on `Makie` for plotting APIs while users choose and load their preferred Makie backend separately.

## Context

- `ext/GLMakieExt.jl` imports `GLMakie` directly and implements `viz_2d`, `viz_3d`, `viz_3d_surface`, `viz_3d_interface_solution`, and `viz_3d_zslice`.
- `Project.toml` declares `GLMakie` as the visualization weak dependency and binds the package extension as `GLMakieExt = ["GLMakie"]`.
- The fallback plotting methods in `src/visualization/viz_2d.jl` and `src/visualization/viz_3d.jl` tell users to load `GLMakie`, which incorrectly implies the package only supports that backend.
- The public visualization API is already exported from `src/BoundaryIntegral.jl`, so the desired change is in dependency wiring and implementation detail rather than in the external API surface.

## Approaches Considered

### 1. Makie-only weak dependency and `MakieExt`

Use `Makie` as the only weak dependency, rename the extension module to `MakieExt`, and implement all plotting helpers against Makie’s frontend API while leaving backend selection to the user.

Pros:
- Keeps `BoundaryIntegral` backend-agnostic.
- Minimizes the code delta from the current extension.
- Matches Makie’s intended split between frontend API and backend packages.

Cons:
- Users must still know to load a Makie backend before rendering.

### 2. Make `Makie` a regular dependency

Move `Makie` from a weak dependency to a regular dependency and keep plotting code always available.

Pros:
- Simplifies extension loading.

Cons:
- Makes every install pay for visualization dependencies even when plotting is unused.
- Broadens the package dependency footprint without a strong need.

### 3. Add backend-specific extensions on top of a Makie frontend

Create a generic `MakieExt` now and reserve separate backend extensions for future backend-specific behavior.

Pros:
- Leaves room for backend-specific customization later.

Cons:
- Adds extra structure and maintenance cost that the current codebase does not need.

## Recommended Design

Adopt approach 1. Rename the extension to `MakieExt`, switch the weak dependency from `GLMakie` to `Makie`, and replace direct `GLMakie` imports with `Makie` imports throughout the extension. The plotting helpers will continue to return `Figure` and mutate Makie axes, but backend activation will stay outside `BoundaryIntegral`.

The public plotting API will remain unchanged. Existing users who currently load `GLMakie` before plotting will continue to work, while users who prefer another Makie backend such as `CairoMakie` will be able to use the same plotting helpers without any package-side backend selection.

## User-Facing Behavior

- `BoundaryIntegral` will no longer tell users to load `GLMakie` specifically.
- Fallback warnings will instead tell users to load `Makie` support by bringing in a Makie backend such as `CairoMakie` or `GLMakie`.
- README installation and visualization guidance will be updated to describe the backend-agnostic workflow.

## Testing

- Add a focused regression test for the fallback warnings so the non-extension behavior points users to Makie plus a backend rather than to `GLMakie` only.
- Update or add a lightweight package metadata test that confirms `Project.toml` now exposes `Makie` and `MakieExt`.
- Avoid backend-specific rendering tests in this refactor unless the existing test environment already supports them cleanly.
