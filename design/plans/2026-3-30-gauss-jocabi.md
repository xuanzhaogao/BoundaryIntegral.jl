# BEM Quadrature for Corner Singularities

## Problem Setup

Two perpendicular boundary segments $\Gamma_1 = \{(s,0): s\in[0,a]\}$ and $\Gamma_2 = \{(0,t): t\in[0,b]\}$ meeting at the origin. The density on $\Gamma_1$ has a known corner singularity $\varphi(s) \sim s^{-1/2}$, and similarly on $\Gamma_2$. The kernel is the normal derivative of the Laplace single-layer potential:

$$K(s,t) = \frac{1}{2\pi}\frac{s}{s^2 + t^2}.$$

Target: assemble the matrix $A_{kj}$ for an FMM-accelerated BIE solver (collocation formulation), with analytic treatment of the singularity and numerical near-field correction.

## Mesh and Basis

Use a graded mesh near the origin on both $\Gamma_1$ and $\Gamma_2$: panel lengths $h_k = \sigma^k h_0$ with $\sigma \approx 0.15$, $k = 0, \dots, K$. Away from the corner, use uniform or mildly graded panels.

On the **innermost panel** $[0, h_K]$ (containing the singular endpoint), represent the density as

$$\varphi(s) = s^{-1/2}\sum_{j=0}^{p-1} c_j\, P_j^{(-1/2,0)}\!\left(\frac{2s}{h_K}-1\right),$$

where $P_j^{(-1/2,0)}$ are Jacobi polynomials orthogonal with respect to $s^{-1/2}$ on $[0, h_K]$. On all other panels, use standard polynomial bases with Gauss-Legendre nodes.

## Far-Field: FMM Evaluation

Discretize the integral $\int_0^a \varphi(s)\,K(s,t)\,ds$ as a sum of point sources.

**Innermost panel** $[0, h_K]$: use $p$-point Gauss-Jacobi quadrature (weight $s^{-1/2}$) with nodes $\{s_i\}$ and weights $\{\hat{w}_i\}$. The FMM charges are

$$q_i = \hat{w}_i \cdot f(s_i), \quad f(s) = s^{1/2}\varphi(s) = \sum_j c_j\,P_j^{(-1/2,0)}\!\left(\frac{2s}{h_K}-1\right).$$

**All other panels**: use $p$-point Gauss-Legendre with nodes $\{s_i\}$, weights $\{w_i\}$. Charges are $q_i = w_i\,\varphi(s_i)$.

Feed all point sources $\{(s_i, 0),\, q_i\}$ into the FMM. The FMM computes

$$A^{\mathrm{far}}\mathbf{c} \approx \sum_i q_i\,K(s_i, t)$$

for all far-field target points $t$ on $\Gamma_2$.

## Collocation and RHS Treatment

The BIE is discretized by collocation: enforce the integral equation at the quadrature nodes $\{t_k\}$ on $\Gamma_2$. The linear system is

$$\sum_j c_j \int_{\Gamma_1} K(s, t_k)\,\phi_j(s)\,ds = g(t_k), \quad k = 1,\dots,N,$$

where $g$ is the boundary data (e.g., Dirichlet data) and $\phi_j$ are the basis functions.

The RHS is simply point evaluation $g(t_k)$ — no projection or inner product is needed. Since $g$ is smooth near the corner, the question is whether the Gauss-Jacobi collocation nodes (chosen to match the singular density basis on $\Gamma_2$) adequately represent $g$.

The answer is yes: $p$-point Gauss-Jacobi$(\alpha,\beta)$ nodes have the same asymptotic distribution as Chebyshev nodes (density $\propto 1/\sqrt{1-x^2}$ on $[-1,1]$), so polynomial interpolation of an analytic function $g$ on these nodes converges at rate $O(\rho^{-p})$ where $\rho > 1$ is determined by the Bernstein ellipse of analyticity of $g$. This rate is independent of the weight parameters $\alpha, \beta$; the difference from Gauss-Legendre is only in the constant prefactor. (Ref: Trefethen, *Approximation Theory and Approximation Practice*, Ch. 19; Szegő, *Orthogonal Polynomials*, §14.)

Therefore, using the same Gauss-Jacobi nodes for both the density basis and the collocation points introduces no order degradation in the RHS representation. This is the cleanest implementation: one set of nodes per panel serves both roles.

## Near-Field Correction

For target points $t_k$ on $\Gamma_2$ that are in the near-field interaction list of the innermost panel, the FMM output is inaccurate. Replace it with a directly computed matrix entry:

$$A_{kj} = \frac{1}{2\pi}\int_0^{h_K} s^{-1/2}\,\frac{s}{s^2+t_k^2}\,P_j^{(-1/2,0)}\!\left(\frac{2s}{h_K}-1\right)ds.$$

The integrand is smooth with respect to the measure $s^{-1/2}ds$, but the kernel $s/(s^2+t_k^2)$ develops a sharp peak of width $O(t_k)$ at $s \sim t_k$ when $t_k \to 0$. Standard Gauss-Jacobi quadrature converges poorly in this regime.

### Option A: Analytic computation via recurrence

Map to $u \in [-1,1]$ via $s = h_K(u+1)/2$, set $\beta = 2t_k/h_K$. The required integrals reduce to the family

$$I_n(\beta) = \int_0^2 \frac{v^{n-1/2}}{v^2+\beta^2}\,dv, \quad n = 0, 1, 2, \dots$$

which satisfies the two-term recurrence

$$I_{n+2}(\beta) = \frac{2^{n+3/2}}{n+3/2} - \beta^2\,I_n(\beta).$$

The initial values $I_0(\beta)$ and $I_1(\beta)$ are expressible in terms of $\arctan$ and $\log$ via the substitution $v = \beta u^2$ and partial fraction decomposition of $1/(u^4+1)$. Matrix entries $A_{kj}$ are then linear combinations of $\{I_n\}$ weighted by the monomial expansion coefficients of $P_j^{(-1/2,0)}$.

### Option B: Numerical computation via regularizing substitution

Apply $s = t_k \tan\theta$, $ds = t_k\sec^2\theta\,d\theta$. Then

$$\frac{s}{s^2+t_k^2}\,ds = \sin\theta\cos\theta\,d\theta,$$

and the integral becomes

$$A_{kj} = \frac{1}{2\pi}\int_0^{\arctan(h_K/t_k)} (t_k\tan\theta)^{-1/2}\,\sin\theta\cos\theta\; P_j^{(-1/2,0)}\!\left(\frac{2t_k\tan\theta}{h_K}-1\right)d\theta.$$

The peak in the kernel is completely absorbed. The integrand is smooth on $[0, \arctan(h_K/t_k)]$ with a mild $\theta^{1/2}$ singularity at $\theta = 0$, handled by either:

1. Gauss-Jacobi on $[0, \Theta]$ with weight $\theta^{1/2}$, or
2. Gauss-Legendre after a further substitution $\theta = \Theta\xi^2$ to regularize the endpoint.

In either case, $O(p)$ quadrature points suffice uniformly in $t_k$.

### Assembly procedure (Option B)

For each near-field target $t_k$:

1. Compute $\Theta_k = \arctan(h_K / t_k)$.
2. Generate $p$-point quadrature nodes $\{\theta_i\}$ and weights $\{w_i^\theta\}$ on $[0, \Theta_k]$.
3. Map back: $s_i = t_k\tan\theta_i$.
4. Evaluate $A_{kj} = \frac{1}{2\pi}\sum_i w_i^\theta\,(s_i)^{-1/2}\,\sin\theta_i\cos\theta_i\;P_j^{(-1/2,0)}\!\left(\frac{2s_i}{h_K}-1\right)$.
5. Subtract the FMM contribution at $t_k$ from the innermost panel and replace with $\sum_j A_{kj}\,c_j$.

## Complexity

| Component | Cost |
|---|---|
| FMM evaluation | $O(N)$ or $O(N\log N)$ |
| Near-field correction per target | $O(p^2)$ ($p$ basis functions $\times$ $p$ quad points) |
| Total near-field | $O(p^2 N_{\mathrm{near}})$, $N_{\mathrm{near}} = O(N)$ |
| Graded mesh DOFs at corner | $O(p\,|\log\epsilon|/|\log\sigma|)$ |