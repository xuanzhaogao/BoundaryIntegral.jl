# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds the BoundaryIntegral.jl module and submodules (`core/`, `kernel/`, `solver/`, `shape/`, `utils/`, `visualization/`).
- `test/` mirrors `src/` with focused test files and a `test/runtests.jl` entry point.
- `docs/` contains Documenter.jl sources and build script.
- `ext/` provides optional integrations (e.g., `GLMakieExt.jl`).

## Build, Test, and Development Commands
- Instantiate deps: `julia --project -e 'using Pkg; Pkg.instantiate()'`
- Run tests: `julia --project -e 'using Pkg; Pkg.test()'`
- Build docs locally: `julia --project=docs docs/make.jl`

## Coding Style & Naming Conventions
- Indentation: 4 spaces, no tabs.
- Types/modules in `CamelCase`, functions and variables in `lowercase_with_underscores`.
- Keep numerical kernels and solvers in `src/kernel/` and `src/solver/` respectively.
- No formatter is configured; follow existing file style and keep changes minimal.

## Testing Guidelines
- Tests use Julia’s `Test` stdlib; group new tests in `test/` and include them from `test/runtests.jl`.
- Name test files by feature area (e.g., `test/kernel/laplace3d.jl`).
- When adding functionality, add or extend a targeted test file and ensure `Pkg.test()` passes.

## Commit & Pull Request Guidelines
- Commit messages follow an imperative, sentence-style summary (e.g., “Add quad_order utility”).
- PRs should include a concise description, rationale, and test evidence (commands + results).
- Link related issues when applicable and update docs/examples if behavior changes.

## Optional: Docs & Visualization Notes
- `GLMakie` support is optional via `ext/GLMakieExt.jl`; guard new visualization features accordingly.
- Keep documentation changes scoped to `docs/src/` and validate with the docs build command.
