# Consistency of `eps_in`/`eps_out` and Normal Direction

## Convention

From `src/core/panels.jl:63`:
> `# in and out is defined according to normal direction of the panel`

The established convention is: **the panel normal points FROM the `eps_in` region INTO the `eps_out` region**. Equivalently, `eps_in` is the permittivity *behind* the normal, `eps_out` is *ahead* of it.

---

## Consistency Check

### 1. External faces (`box3d_multi.jl:343`)
```julia
push!(regions, (ra, rb, rc, rd, fn, epses[box_id], eps_out, ie, ic))
#                              ↑           ↑            ↑
#                           fn = outward  eps_in=box   eps_out=background
```
`fn` = outward normal of the box face → points FROM box interior TOWARD exterior background.  
`eps_in = epses[box_id]` (behind the normal = box interior) ✓  
`eps_out = background` (ahead of the normal = exterior) ✓

---

### 2. Shared faces (`box3d_multi.jl:134–154, 351`)
```julia
# stored normal = -n1 = points FROM id_hi TOWARD id_lo
push!(shared, (region, i, j, (-n1[1], -n1[2], -n1[3])))
# id_lo=i, id_hi=j

push!(regions, (a, b, c, d, normal, epses[id_hi], epses[id_lo], ...))
#                                         ↑              ↑
#                                    eps_in=j          eps_out=i
```
Normal points FROM box j (id_hi) TOWARD box i (id_lo).  
`eps_in = epses[id_hi]` = box j permittivity (behind the normal) ✓  
`eps_out = epses[id_lo]` = box i permittivity (ahead of the normal) ✓  

**Note:** the "inside" direction on a shared face is always assigned to the higher-indexed box. This is an arbitrary but consistently applied tie-breaking rule.

---

### 3. LHS operator (`dielectric_box3d.jl:10`)
```julia
t = 0.5 * (eps_out + eps_in) / (eps_out - eps_in)
```
Uses the `eps_in`/`eps_out` stored per-panel, so it sees the same convention ✓.

---

### 4. DT kernel (`laplace3d.jl:9–14, 36`)
```julia
# laplace3d_grad(src, trg, norm) = norm · (trg - src) / (4π |trg - src|³)
DT[i,j] = laplace3d_grad(x_j, x_i, n_i) * w_j
         = n_i · (x_i - x_j) / (4π |x_i - x_j|³) * w_j
```
The kernel uses `n_i`, the panel normal at the *target* point, consistent with the standard Neumann-Poincaré (double-layer transpose) operator ✓.

---

### 5. RHS (`dielectric_box3d.jl:95`)
```julia
Rhs[i] = -q * laplace3d_grad(x_src, x_i, n_i) / eps_src
```
Same normal convention, consistent with LHS ✓.

---

## Summary

The `eps_in`/`eps_out` assignment and normal direction are **fully consistent** throughout:

| Location | Normal direction | `eps_in` | `eps_out` |
|---|---|---|---|
| External box face | Outward from box | Box permittivity | Background |
| Shared interface | From id_hi to id_lo | `epses[id_hi]` | `epses[id_lo]` |
| LHS diagonal | (reads stored per-panel) | ✓ | ✓ |
| DT kernel | Target panel normal `n_i` | ✓ | ✓ |
| RHS kernel | Target panel normal `n_i` | ✓ | ✓ |

One thing worth being aware of: for a shared face between box `i` and box `j` with `i < j`, the normal always points from `j` toward `i` and `eps_in = epses[j]`. There is no physical reason for this ordering — it is a symmetric tie-breaking convention that is applied uniformly and does not affect correctness.
