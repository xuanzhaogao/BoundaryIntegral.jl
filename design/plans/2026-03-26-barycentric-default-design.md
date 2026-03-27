# Barycentric Default Design

Goal: replace the Horner/monomial default interpolation path with cached barycentric weights in the hot paths because barycentric evaluation is at least as fast and numerically safer at higher quadrature order.

Design:
- Cache barycentric weights on `FlatPanel` during panel construction.
- Make `interface_approx` use `barycentric_row!` with preallocated work buffers.
- Make the `laplace3d_near` hcubature and prolongation paths use `barycentric_row!` as the default evaluator.
- Keep the change scoped to the current Horner-default call sites so the existing allocation wins remain in place.
- Add a regression test that enforces exact reproduction of high-order polynomial data through `interface_approx`.

Verification:
- Run the targeted interpolation and near-field tests.
- Run the full `Pkg.test()` suite with `BI_RUN_FULL_TESTS=1`.
- Re-run the existing benchmark harness on the updated tree and compare against the pre-switch measurements from this session.
