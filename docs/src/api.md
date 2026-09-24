```@meta
CurrentModule = BoundaryIntegral
```

# API reference

```@index
```

```@autodocs
Modules = [BoundaryIntegral]
```

## Internals: truncated-kernel method

`BoundaryIntegral.TKM3D` evaluates the free-space Laplace potential of volume sources
with a truncated-kernel spectral method. It is used internally by the volume-source
right-hand sides and by [`PrecomputedVolumeField`](@ref).

```@autodocs
Modules = [BoundaryIntegral.TKM3D]
```
