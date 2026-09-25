# Development

Clone the repository and instantiate its dependencies:

```sh
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Tests

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

The default run covers the fast tests. Set `BI_RUN_FULL_TESTS=1` for the full suite,
including the 3D near-correction, box geometry, and solver tests:

```sh
BI_RUN_FULL_TESTS=1 julia --project -e 'using Pkg; Pkg.test()'
```

## Documentation

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The built site is written to `docs/build/`.
