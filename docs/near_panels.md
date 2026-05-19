# Near-Singular Quadrature Correction in 3D Laplace BIE

## 1. Problem

In a 3D Laplace boundary integral equation (BIE), one often needs to evaluate layer potentials of the form

\[
u(x) = \int_{\Gamma} K(x,y)\sigma(y)\,dS_y,
\]

where \(K(x,y)\) is a Laplace kernel, for example

\[
G(x,y)=\frac{1}{4\pi |x-y|}
\]

for the single-layer potential, or

\[
\partial_{n_y}G(x,y)
=
\frac{n_y\cdot (x-y)}{4\pi |x-y|^3}
\]

for the double-layer potential.

When the target point \(x\) is close to, but not necessarily on, a source patch \(\Gamma_j\), the integrand becomes nearly singular. Standard smooth quadrature rules can lose accuracy severely. Therefore, special near-field quadrature correction is required.

The goal is to compute

\[
\int_{\Gamma_j} K(x,y)\sigma(y)\,dS_y
\]

accurately for near target-patch pairs \((x,\Gamma_j)\).

---

## 2. Patch Representation

Assume the surface is decomposed into high-order patches. A patch \(\Gamma_j\) is represented by a smooth parametrization

\[
X^j:T_0\to \Gamma_j,
\]

where \(T_0\) is a reference triangle.

The near integral becomes

\[
\int_{\Gamma_j} K(x,y)\sigma(y)\,dS_y
=
\int_{T_0}
K(x,X^j(u,v))
\sigma(u,v)
J^j(u,v)
\,du\,dv,
\]

where \(J^j(u,v)\) is the surface Jacobian.

---

## 3. Polynomial Representation of the Density

The method assumes that the discrete density on each patch is represented in a local polynomial space. For a patch of order \(p\),

\[
\sigma(u,v)
\approx
\sum_{n+m<p} s_{nm} K_{nm}(u,v),
\]

where \(K_{nm}\) are local orthogonal polynomial basis functions, such as Koornwinder polynomials on the reference triangle.

The coefficients \(s_{nm}\) are not obtained by adaptive integration. They are obtained from the nodal density values by a fixed values-to-coefficients map:

\[
s_{nm}
=
\sum_{\ell=1}^{n_p}
V_{(nm),\ell}\sigma_\ell.
\]

Here:

- \(\sigma_\ell\) are the density values at the original patch nodes;
- \(V\) is a precomputed matrix mapping nodal values to polynomial coefficients;
- \(n_p\) is the number of nodes on the patch.

Thus the density is treated as an element of the local discrete polynomial space.

---

## 4. Kernel-Weighted Polynomial Moments

Substituting the polynomial expansion into the near integral gives

\[
\int_{\Gamma_j} K(x,y)\sigma(y)\,dS_y
\approx
\sum_{n+m<p}
s_{nm}
I^j_{nm}(x),
\]

where

\[
I^j_{nm}(x)
=
\int_{T_0}
K(x,X^j(u,v))
K_{nm}(u,v)
J^j(u,v)
\,du\,dv.
\]

The quantities \(I^j_{nm}(x)\) are the key objects computed by near correction.

They are kernel-weighted polynomial moments. For each near target \(x\) and source patch \(\Gamma_j\), one computes these moments accurately.

This is the expensive part.

---

## 5. Adaptive Computation of the Moments

For a near target-patch pair, standard quadrature may fail because

\[
K(x,X^j(u,v))
\]

varies sharply over the reference triangle.

Therefore, the moment integrals

\[
I^j_{nm}(x)
\]

are computed by adaptive quadrature.

A typical adaptive strategy is:

1. Apply a high-order quadrature rule on the current triangle.
2. Subdivide the triangle into four child triangles.
3. Apply the same quadrature rule on the children.
4. Compare the parent integral with the sum of child integrals.
5. Refine recursively until the estimated error is below tolerance.

The adaptive integrand is

\[
K(x,X^j(u,v))
K_{nm}(u,v)
J^j(u,v).
\]

This is done for all basis functions \(K_{nm}\) in the local polynomial space.

The computation is brute-force in spirit, but it is performed only during precomputation of the near-field correction, not during every matrix-vector product.

---

## 6. Construction of Corrected Near Weights

Once the moments \(I^j_{nm}(x)\) are computed, the near integral can be written directly in terms of nodal density values.

Using

\[
s_{nm}
=
\sum_{\ell=1}^{n_p}
V_{(nm),\ell}\sigma_\ell,
\]

we get

\[
\int_{\Gamma_j} K(x,y)\sigma(y)\,dS_y
\approx
\sum_{n+m<p}
I^j_{nm}(x)
\sum_{\ell=1}^{n_p}
V_{(nm),\ell}\sigma_\ell.
\]

Rearranging,

\[
\int_{\Gamma_j} K(x,y)\sigma(y)\,dS_y
\approx
\sum_{\ell=1}^{n_p}
a^j_\ell(x)\sigma_\ell,
\]

where

\[
a^j_\ell(x)
=
\sum_{n+m<p}
I^j_{nm}(x)
V_{(nm),\ell}.
\]

The vector

\[
a^j(x)
=
\{a^j_\ell(x)\}_{\ell=1}^{n_p}
\]

is the corrected quadrature row for the target \(x\) and patch \(\Gamma_j\).

Thus the near correction converts the near integral into a small dense local matrix row acting on the nodal density values.

---

## 7. Reuse of Information

The method is efficient because much of the expensive information can be reused.

### 7.1 Reuse across matrix-vector products

The corrected near weights \(a^j_\ell(x)\) depend on:

- the geometry of the patch;
- the target location;
- the kernel;
- the local polynomial basis;
- the requested quadrature accuracy.

They do not depend on the current density values.

Therefore, once these weights are precomputed, each later matrix-vector product only requires

\[
\sum_{\ell=1}^{n_p} a^j_\ell(x)\sigma_\ell.
\]

No adaptive quadrature is needed during the iterative solve.

---

### 7.2 Reuse across basis functions

For fixed target \(x\) and patch \(\Gamma_j\), all moments

\[
I^j_{nm}(x)
\]

share the same geometry values:

\[
X^j(u,v), \qquad J^j(u,v),
\]

and the same kernel values:

\[
K(x,X^j(u,v)).
\]

Only the polynomial factor \(K_{nm}(u,v)\) changes.

Therefore, during adaptive quadrature, one can evaluate the geometry and kernel once at quadrature nodes and reuse them for all polynomial basis functions.

---

### 7.3 Reuse across targets near the same patch

Many target points may be near the same source patch. The patch geometry, polynomial basis values, Jacobians, and subdivision structure can be reused.

For a fixed patch \(\Gamma_j\), quantities such as

\[
X^j(u,v),\qquad J^j(u,v),\qquad K_{nm}(u,v)
\]

on adaptive quadrature nodes are independent of the target, except for the kernel factor

\[
K(x,X^j(u,v)).
\]

Thus source-side data can be cached per patch and reused across many nearby targets.

---

### 7.4 Reuse through sparse near-field structure

Only a small subset of target-patch pairs are near. Therefore, the corrected quadrature rows are stored in a sparse structure.

For each target \(x_i\), one stores the list of nearby patches \(\Gamma_j\). For each such pair, one stores the corresponding local corrected weights

\[
a^j_\ell(x_i).
\]

During application, the near correction is a sparse local accumulation over near target-patch pairs.

---

## 8. Combination with FMM

In fast BIE solvers, the total potential is often evaluated as

\[
u
=
u^{\mathrm{FMM}}
+
u^{\mathrm{near,corr}}
-
u^{\mathrm{near,over}}.
\]

Here:

- \(u^{\mathrm{FMM}}\) is computed using oversampled smooth quadrature and accelerated by FMM;
- \(u^{\mathrm{near,corr}}\) is the accurate locally corrected near contribution;
- \(u^{\mathrm{near,over}}\) subtracts the inaccurate oversampled contribution from near patches that was already included in the FMM result.

The subtraction is necessary because the FMM evaluation includes all source nodes, including near ones. Near interactions are therefore first counted inaccurately by the oversampled FMM quadrature and then replaced by corrected quadrature.

---

## 9. Interpretation

The essential idea is:

1. Approximate the density on each patch by a local polynomial interpolant.
2. For each near target-patch pair, accurately compute the action of the kernel on every polynomial basis function.
3. Convert these polynomial moments into corrected weights acting on nodal density values.
4. Store the resulting weights.
5. Reuse them in every matrix-vector product.

Thus the method is not directly integrating the density at solve time. Instead, it precomputes the near evaluation functional on the local discrete polynomial space.

In formula form:

\[
\boxed{
I^j_{nm}(x)
=
\int_{T_0}
K(x,X^j(u,v))
K_{nm}(u,v)
J^j(u,v)
\,du\,dv
}
\]

followed by

\[
\boxed{
a^j_\ell(x)
=
\sum_{n+m<p}
I^j_{nm}(x)
V_{(nm),\ell}
}
\]

and finally

\[
\boxed{
\int_{\Gamma_j} K(x,y)\sigma(y)\,dS_y
\approx
\sum_{\ell=1}^{n_p}
a^j_\ell(x)\sigma_\ell.
}
\]

This is the core near-singular quadrature correction mechanism.

---

## 10. Main Takeaway

The near correction is based on the fact that the numerical density lives in a finite-dimensional local polynomial space. The method computes, by adaptive quadrature or specialized generalized Gaussian quadrature, the kernel-weighted integrals of each basis polynomial. These are then assembled into reusable corrected quadrature weights.

The expensive adaptive integration is moved into a precomputation stage. The repeated solve stage only applies a sparse near-correction matrix.
