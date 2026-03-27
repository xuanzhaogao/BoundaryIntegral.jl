# BoundaryIntegral.jl Cleanup & Refactor Plan

**Goal:** Improve repository hygiene, code organization, and maintainability without changing any behavior.

**Guiding principles:**
- No behavioral changes. All 305 tests must pass after every step.
- Small, reviewable commits.
- Safe cleanup before structural refactors.
- No over-engineering.

---

## Phase 1: Safe Cleanup (no structural changes)

### Step 1: Remove dead code and fix typos

**Objective:** Delete commented-out code and fix comment typos.

**Files affected:**
- `src/kernel/laplace2d.jl` — remove lines 167-201 (commented-out `laplace2d_gradtrg_fmm2d`)
- `src/shape/box3d.jl` — fix typo line 1: "temporary panal" → "temporary panel"
- `src/shape/box2d.jl` — update comment line 1 to not say "temporary" (it's a permanent internal type)
- `src/shape/box3d.jl` — same for line 1

**Expected benefit:** Less noise, no misleading comments.
**Risks:** None.
**Validation:** `julia --project -e 'using Pkg; Pkg.test()'` — 305 tests pass.

---

### Step 2: Remove unused dependencies

**Objective:** Remove `Roots` and `SparseMatricesCOO` from Project.toml (zero usage in codebase). Replace `Statistics.mean` with inline `sum/length` to drop `Statistics`.

**Files affected:**
- `Project.toml` — remove `Roots`, `SparseMatricesCOO`, `Statistics` from `[deps]` and `[compat]`
- `src/BoundaryIntegral.jl` — remove `Statistics` and `Roots` from `using` line
- `src/shape/box3d.jl` — replace `mean(abs.(vs.weights))` with `sum(abs, vs.weights) / length(vs.weights)`

**Expected benefit:** Faster install, cleaner dependency tree.
**Risks:** Low. Verify `Roots` isn't used by any transitive path.
**Validation:** `Pkg.instantiate()` + `Pkg.test()` — 305 tests pass.

---

### Step 3: Clean up docs directory

**Objective:** Move internal design docs out of `docs/` (which is for Documenter.jl user docs). Consolidate design docs under a single top-level `design/` directory.

**Files affected:**
- Move `docs/plans/` → `design/plans/`
- Move `docs/superpowers/` → `design/superpowers/`
- Keep `docs/make.jl` and `docs/src/` (Documenter.jl)

**Expected benefit:** Clear separation: `docs/` = user docs, `design/` = internal development notes.
**Risks:** None (no code changes).
**Validation:** `Pkg.test()` still passes. Verify `docs/make.jl` still works.

---

## Phase 2: Structural Refactors

### Step 4: Split box3d.jl into focused files

**Objective:** Break the 1437-line `src/shape/box3d.jl` into logical subfiles.

**Proposed split:**
- `src/shape/box3d.jl` — keep `TempPanel3D`, `rect_panel3d_discretize`, `divide_temp_panel3d`, `rect_panel3d_adaptive_panels` (core geometry, ~135 lines)
- `src/shape/box3d_single.jl` — `_box3d_geometry`, `_box3d_face_quads`, `single_dielectric_box3d` (single-box builder, ~50 lines)
- `src/shape/box3d_multi.jl` — `_box3d_faces_at_center`, `_rect_overlap_3d`, `_detect_shared_faces_3d`, `_subtract_rects_from_face_3d`, `_multi_box3d_face_regions`, `multi_dielectric_box3d`, `multi_dielectric_box3d_rhs_adaptive` (multi-box, ~270 lines)
- `src/shape/box3d_rhs_adaptive.jl` — `rhs_panel3d_integral`, `rhs_panel3d_resolved`, `rect_panel3d_rhs_adaptive_panels` (both overloads), `rect_panel3d_rhs_adaptive_panels_varquad`, `single_dielectric_box3d_rhs_adaptive` (all overloads), `single_dielectric_box3d_rhs_adaptive_varquad` (all overloads) (~420 lines)
- `src/shape/box3d_fmm_helpers.jl` — `_volume_source_fmm_sources`, `_estimate_source_spacing`, `_estimate_tkm3dc_kmax`, `_classify_near_far_panels`, `_classify_near_far_targets`, `_rhs_from_grad`, `_rhs_volume_targets_hybrid`, `_rhs_panel3d_refinement_targets`, `_rhs_panel3d_resolved_volume_fmm`, `_box3d_rhs_adaptive_initial_panels`, `best_grid_mn`-related helpers (~400 lines)

**Files affected:**
- `src/shape/box3d.jl` — shrink to core geometry only
- Create 4 new files in `src/shape/`
- `src/BoundaryIntegral.jl` — add `include()` lines for new files

**Expected benefit:** Each file has one clear responsibility, fits in context, easy to navigate.
**Risks:** Medium — must preserve include order (functions must be defined before use). Must verify no circular dependencies.
**Validation:** `Pkg.test()` — 305 tests pass. `using BoundaryIntegral` loads without error.

---

### Step 5: Split laplace3d_near.jl into focused files

**Objective:** Break the 905-line `src/kernel/laplace3d_near.jl` into logical subfiles.

**Proposed split:**
- `src/kernel/laplace3d_near.jl` — keep neighbor list building, prolongation matrix, and the main corrected DT/D/pottrg functions (~350 lines)
- `src/kernel/laplace3d_near_upsampling.jl` — upsampling panel refinement, local correction matrices (~300 lines)
- `src/kernel/laplace3d_near_hcubature.jl` — hcubature-based correction variants (~250 lines)

**Files affected:**
- `src/kernel/laplace3d_near.jl` — shrink
- Create 2 new files in `src/kernel/`
- `src/BoundaryIntegral.jl` — add includes

**Expected benefit:** Easier to understand each correction strategy independently.
**Risks:** Medium — same include-order concern.
**Validation:** `Pkg.test()` — 305 tests pass, including full tests with `BI_RUN_FULL_TESTS=1`.

---

### Step 6: Rename solver functions to follow Julia conventions

**Objective:** Rename `Lhs_dielectric_box*` → `lhs_dielectric_box*` and `Rhs_dielectric_box*` → `rhs_dielectric_box*` to follow snake_case convention per AGENTS.md. Keep old names as deprecated aliases.

**Files affected:**
- `src/solver/dielectric_box2d.jl` — rename functions
- `src/solver/dielectric_box3d.jl` — rename functions
- `src/BoundaryIntegral.jl` — update exports, add deprecation aliases
- `test/solver/dielectric_box2d.jl` — update calls
- `test/solver/dielectric_box3d.jl` — update calls
- `test/shape/multi_box3d.jl` — update calls

**Expected benefit:** Consistent naming per AGENTS.md guidelines.
**Risks:** Low-Medium — breaking change for downstream users. Mitigated by keeping CamelCase aliases with `Base.@deprecate`.

**FLAGGED AS RISKY:** This is a public API change. Even with deprecation aliases, downstream code may break if not updated. Recommend confirming with user before execution.

**Validation:** `Pkg.test()` — 305 tests pass with new names. Verify old names still work via deprecation.

---

## Phase 3: Optional Nice-to-Have

### Step 7: Add missing test for multi_box3d with different-sized boxes

**Objective:** Add a permanent test for different-sized touching boxes (the case we manually verified but didn't commit as a test).

**Files affected:**
- `test/shape/multi_box3d.jl` — add test

**Expected benefit:** Regression protection for partial-face overlap.
**Risks:** None.
**Validation:** Test count increases, all pass.

---

### Step 8: Consolidate tiny utility files

**Objective:** `linear_algebra.jl` (5 lines), `bernstein.jl` (22 lines), and `gaussians.jl` (21 lines) are very small. Consider merging `linear_algebra.jl` and `gaussians.jl` into relevant neighbors, but only if it improves clarity.

**Decision:** Leave `bernstein.jl` alone (self-contained math). Leave `gaussians.jl` alone (distinct purpose). Leave `linear_algebra.jl` alone (it's the solve wrappers, logically separate even if tiny). **Skip this step** — the current split is fine, tiny files are not a problem.

---

## Summary Table

| Step | Type | Risk | Priority |
|------|------|------|----------|
| 1. Remove dead code/typos | Safe cleanup | None | HIGH |
| 2. Remove unused deps | Safe cleanup | Low | HIGH |
| 3. Clean up docs dir | Safe cleanup | None | HIGH |
| 4. Split box3d.jl | Structural refactor | Medium | HIGH |
| 5. Split laplace3d_near.jl | Structural refactor | Medium | MEDIUM |
| 6. Rename Lhs_/Rhs_ functions | Structural refactor | **RISKY** | LOW (needs approval) |
| 7. Add different-size box test | Nice-to-have | None | LOW |
