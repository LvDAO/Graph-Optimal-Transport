# AGENT TASK: Implement harmonic-mean graph dynamic OT warm start as a MOSEK Fusion SOCP

## 0. Goal

Implement a **harmonic-mean** discrete dynamic optimal transport (OT) solver as a **second-order cone program (SOCP)** in **MOSEK Fusion**.

This SOCP is **not** the final logarithmic-mean solver.
Its purpose is to compute a **robust warm start** for the custom solver.

The output of this SOCP should provide at least:
- `rho[i, x]`
- `m[i, e]`
- `rhobar[i, x]`

and then reconstruct warm-start auxiliary fields for the custom solver:
- `q[i, x] = rhobar[i, x]`
- `rho_minus[i, e=(x,y)] = rhobar[i, x]`
- `rho_plus[i, e=(x,y)]  = rhobar[i, y]`
- `vartheta[i, e=(x,y)]  = 2 * rhobar[i, x] * rhobar[i, y] / (rhobar[i, x] + rhobar[i, y])`
  with the convention `vartheta = 0` if both endpoints are zero.

---

## 1. Data and indexing

We work on a finite reversible graph:
- node set: `X = {0, ..., n-1}`
- directed arc set:
  `S = {(x,y) : Q(x,y) > 0}`

Inputs:
- `n`: number of nodes
- `a = |S|`: number of directed arcs
- `N`: number of time intervals
- `h = 1.0 / N`
- `pi[x] > 0`: reference measure on nodes
- `Q[x,y] >= 0`: transition weight / edge weight
- `rhoA[x]`, `rhoB[x]`: endpoint densities

Use a directed-arc representation:
- each arc index `e` corresponds to one ordered pair `(src[e], dst[e])`
- only create variables for arcs in `S`

Precompute the edge objective weights:
- `w[e] = Q[src[e], dst[e]] * pi[src[e]]`

---

## 2. Decision variables

Use the following variables.

### 2.1 Main variables
- `rho[i, x]` for `i = 0..N`, `x = 0..n-1`
  - node density at time node `i`
- `rhobar[i, x]` for `i = 0..N-1`, `x = 0..n-1`
  - time-average density on interval `i`
- `m[i, e]` for `i = 0..N-1`, `e = 0..a-1`
  - directed momentum / flux on arc `e` during interval `i`

### 2.2 SOCP epigraph variables
For each interval `i` and arc `e=(x,y)` introduce:
- `u[i, e] >= 0`
- `v[i, e] >= 0`

They encode the harmonic-mean action via rotated quadratic cones.

---

## 3. Harmonic-mean action reformulation

For harmonic mean,
\[
\theta_{\mathrm{har}}(s,t) = \frac{2st}{s+t}.
\]

Therefore
\[
\frac{m^2}{\theta_{\mathrm{har}}(s,t)}
=
\frac{m^2(s+t)}{2st}
=
\frac{m^2}{2s} + \frac{m^2}{2t}.
\]

At time interval `i` and arc `e=(x,y)`, define
- `s = rhobar[i, x]`
- `t = rhobar[i, y]`

Then the local cost is
\[
\frac{m[i,e]^2}{2\,rhobar[i,x]}
+
\frac{m[i,e]^2}{2\,rhobar[i,y]}.
\]

Introduce `u[i,e]`, `v[i,e]` such that
\[
u[i,e] \ge \frac{m[i,e]^2}{2\,rhobar[i,x]},
\qquad
v[i,e] \ge \frac{m[i,e]^2}{2\,rhobar[i,y]}.
\]

These are equivalent to the rotated cone constraints
\[
(rhobar[i,x],\; u[i,e],\; m[i,e]) \in \mathcal Q_r^3,
\]
\[
(rhobar[i,y],\; v[i,e],\; m[i,e]) \in \mathcal Q_r^3,
\]
where
\[
\mathcal Q_r^3 = \{(a,b,c): 2ab \ge c^2,\ a\ge 0,\ b\ge 0\}.
\]

---

## 4. Full SOCP formulation

### 4.1 Objective
Minimize
\[
\frac{h}{2}
\sum_{i=0}^{N-1}\sum_{e=(x,y)\in S}
w[e]\,(u[i,e] + v[i,e]).
\]

### 4.2 Boundary constraints
\[
rho[0,x] = rhoA[x], \qquad rho[N,x] = rhoB[x]
\quad \forall x.
\]

### 4.3 Time-average constraints
\[
2\,rhobar[i,x] = rho[i,x] + rho[i+1,x]
\quad \forall i=0,\dots,N-1,\ \forall x.
\]

### 4.4 Discrete continuity constraints
For every interval `i` and node `x`,
\[
\frac{rho[i+1,x] - rho[i,x]}{h}
+
\frac12
\sum_{y:\,(x,y)\in S} Q(x,y)\,(-m[i,(x,y)])
+
\frac12
\sum_{y:\,(y,x)\in S} Q(x,y)\,m[i,(y,x)]
= 0.
\]

Equivalent compact form:
\[
\frac{rho[i+1] - rho[i]}{h} + C\,m[i] = 0,
\]
where `C in R^{n x a}` is the precomputed discrete divergence / continuity matrix.

Construct `C` so that:
- if arc `e=(x,y)`, then
  - `C[x,e] += -0.5 * Q[x,y]`
  - `C[y,e] += +0.5 * Q[x,y]`
- all other entries zero

This matches
\[
(Cm)_x = \frac12 \sum_y Q(x,y)\bigl(m(y,x)-m(x,y)\bigr).
\]

### 4.5 Positivity
\[
rho[i,x] \ge 0,\quad rhobar[i,x] \ge 0,\quad u[i,e] \ge 0,\quad v[i,e] \ge 0.
\]

### 4.6 Rotated cone constraints
For each interval `i` and arc `e=(x,y)`:
\[
(rhobar[i,x],\,u[i,e],\,m[i,e]) \in \mathcal Q_r^3,
\]
\[
(rhobar[i,y],\,v[i,e],\,m[i,e]) \in \mathcal Q_r^3.
\]

---

## 5. MOSEK Fusion implementation plan

## 5.1 Official MOSEK API pages

Primary documentation pages to read before implementation:

### Python Fusion
- Conic quadratic tutorial:
  https://docs.mosek.com/latest/pythonfusion/tutorial-cqo-shared.html
- Accessing solution:
  https://docs.mosek.com/latest/pythonfusion/accessing-solution.html
- Fusion conic modeling overview:
  https://docs.mosek.com/latest/pythonfusion/modeling.html
- Parametrization / reoptimization:
  https://docs.mosek.com/latest/pythonfusion/tutorial-parametrization.html
- Problem modification / reoptimization:
  https://docs.mosek.com/latest/pythonfusion/tutorial-reoptimization.html
- License setup:
  https://docs.mosek.com/latest/licensing/client-setup.html

### C++ Fusion
- Conic quadratic tutorial:
  https://docs.mosek.com/latest/cxxfusion/tutorial-cqo-shared.html
- C++ Fusion introduction:
  https://docs.mosek.com/latest/cxxfusion/index.html

---

## 5.2 API summary the implementation should follow

### Model structure
Use:
- `Model(...)`
- `M.variable(...)`
- `M.constraint(...)`
- `M.objective(...)`
- `M.solve()`

Fusion modeling rule:
- every constraint is of the form
  `Expression belongs to Domain`

### Variable domains
Use:
- `Domain.greaterThan(0.0)` for nonnegative variables
- `Domain.unbounded()` for free variables
- `Domain.equalsTo(0.0)` for equality constraints
- `Domain.inRotatedQCone()` for rotated SOC constraints

### Useful expression builders
Use:
- `Expr.add(...)`
- `Expr.sub(...)`
- `Expr.mul(...)`
- `Expr.dot(...)`
- `Expr.sum(...)`
- `Expr.flatten(...)`
- `Var.vstack(...)` or `Expr.vstack(...)`

### Solution extraction
After `M.solve()` use:
- `rho.level()`
- `rhobar.level()`
- `m.level()`
- optionally `constraint.level()` and `constraint.dual()`

### Repeated solves
If solving many OT problems with the same graph / same sparsity pattern:
- prefer **fixed structure**
- optionally use `Model.parameter(...)` for data that changes between solves
- candidates for parameters:
  - `rhoA`
  - `rhoB`
  - maybe objective weights if needed
- this allows repeated solves without rebuilding the entire model

### License
MOSEK should find the license at:
- Linux/macOS: `$HOME/mosek/mosek.lic`
- or via environment variable:
  `MOSEKLM_LICENSE_FILE=/absolute/path/to/mosek.lic`

---

## 6. Suggested implementation order

### Step 1: Build sparse graph objects
Create:
- `src[a]`
- `dst[a]`
- `w[a] = Q[src[a],dst[a]] * pi[src[a]]`
- sparse continuity matrix `C`

### Step 2: Create variables
Recommended Python Fusion shapes:
- `rho    : [N+1, n]`
- `rhobar : [N,   n]`
- `m      : [N,   a]`
- `u      : [N,   a]`
- `v      : [N,   a]`

### Step 3: Add boundary constraints
Fix first and last time slices of `rho`.

### Step 4: Add averaging constraints
For all `i,x`:
- `2*rhobar[i,x] = rho[i,x] + rho[i+1,x]`

### Step 5: Add continuity constraints
For each `i`:
\[
(rho[i+1,:]-rho[i,:])/h + C @ m[i,:] = 0.
\]

### Step 6: Add rotated cone constraints
For each `i,e=(x,y)`:
- `Var.vstack(rhobar[i,x], u[i,e], m[i,e]) in RotatedQCone`
- `Var.vstack(rhobar[i,y], v[i,e], m[i,e]) in RotatedQCone`

### Step 7: Add objective
\[
\frac h2 \sum_{i,e} w[e](u[i,e]+v[i,e]).
\]

### Step 8: Solve and export warm start
Export:
- `rho`
- `rhobar`
- `m`

Then reconstruct:
- `q`
- `rho_minus`
- `rho_plus`
- `vartheta_har`

---

## 7. Warm-start mapping to the custom solver

Given the SOCP solution:

### 7.1 Primary fields
- `rho_init = rho`
- `m_init = m`
- `rho_bar_init = rhobar`

### 7.2 Reconstruct helper fields
For each interval `i` and node `x`:
- `q_init[i,x] = rhobar[i,x]`

For each interval `i` and directed arc `e=(x,y)`:
- `rho_minus_init[i,e] = rhobar[i,x]`
- `rho_plus_init[i,e]  = rhobar[i,y]`

### 7.3 Harmonic mean initialization of vartheta
For each interval `i` and directed arc `e=(x,y)`:
\[
vartheta_{init}[i,e]
=
\begin{cases}
\frac{2\,rhobar[i,x]\,rhobar[i,y]}{rhobar[i,x]+rhobar[i,y]},
& rhobar[i,x]+rhobar[i,y] > 0,\\
0, & \text{otherwise.}
\end{cases}
\]

This gives a consistent harmonic-mean warm start for the later logarithmic-mean solver.

---

## 8. Important implementation notes

1. Do **not** implement the Erbar splitting slack variables for the SOCP itself.
   For harmonic mean SOCP we only need:
   - `rho`
   - `rhobar`
   - `m`
   - `u`
   - `v`

2. Keep the first implementation on the **full directed arc set**.
   Do **not** do antisymmetry reduction yet.
   First get correctness, then optimize.

3. Build the continuity operator once and reuse it.

4. Store the solved warm start in a file format directly usable by the custom solver:
   - `.npz`
   - or HDF5
   - or a structured JSON if small

5. Add a verification routine after solve:
   - endpoint residual
   - continuity residual
   - minimum rho
   - objective value

---

## 9. Expected deliverables

Implement the following:

### 9.1 Core builder
A function like
```python
build_harmonic_socp_model(graph, rhoA, rhoB, N) -> (model, handles)
````

where `handles` contains references to:

* `rho`
* `rhobar`
* `m`
* `u`
* `v`

### 9.2 Solver wrapper

A function like

```python
solve_harmonic_socp_warm_start(graph, rhoA, rhoB, N) -> dict
```

returning:

* `rho`
* `m`
* `rho_bar`
* `q`
* `rho_minus`
* `rho_plus`
* `vartheta_har`
* diagnostics

### 9.3 Diagnostics

Return:

* objective value
* continuity max residual
* endpoint max residual
* min rho
* solve status

---

## 10. Minimal MOSEK Fusion pseudocode skeleton

```python
from mosek.fusion import *
import numpy as np

def solve_harmonic_socp_warm_start(graph, rhoA, rhoB, N):
    n = graph.num_nodes
    src = graph.src
    dst = graph.dst
    w = graph.edge_weights   # w[e] = Q[src[e],dst[e]] * pi[src[e]]
    a = len(src)
    h = 1.0 / N
    C = graph.continuity_matrix  # shape (n, a)

    with Model("harmonic_ot_socp") as M:
        rho    = M.variable("rho",    [N+1, n], Domain.greaterThan(0.0))
        rhobar = M.variable("rhobar", [N,   n], Domain.greaterThan(0.0))
        m      = M.variable("m",      [N,   a], Domain.unbounded())
        u      = M.variable("u",      [N,   a], Domain.greaterThan(0.0))
        v      = M.variable("v",      [N,   a], Domain.greaterThan(0.0))

        # boundary constraints
        M.constraint(rho.slice([0,0], [1,n]), Domain.equalsTo(np.asarray(rhoA).reshape(1,n)))
        M.constraint(rho.slice([N,0], [N+1,n]), Domain.equalsTo(np.asarray(rhoB).reshape(1,n)))

        # averaging constraints
        for i in range(N):
            M.constraint(
                Expr.sub(
                    Expr.mul(2.0, rhobar.slice([i,0], [i+1,n])),
                    Expr.add(rho.slice([i,0], [i+1,n]),
                             rho.slice([i+1,0], [i+2,n]))
                ),
                Domain.equalsTo(0.0)
            )

        # continuity constraints
        for i in range(N):
            M.constraint(
                Expr.add(
                    Expr.mul(1.0/h,
                             Expr.sub(rho.slice([i+1,0],[i+2,n]),
                                      rho.slice([i,0],[i+1,n]))),
                    Expr.mul(C, m.slice([i,0],[i+1,a]))
                ),
                Domain.equalsTo(0.0)
            )

        # rotated cones
        for i in range(N):
            for e in range(a):
                x = src[e]
                y = dst[e]
                M.constraint(
                    Var.vstack(rhobar.index(i, x), u.index(i, e), m.index(i, e)),
                    Domain.inRotatedQCone()
                )
                M.constraint(
                    Var.vstack(rhobar.index(i, y), v.index(i, e), m.index(i, e)),
                    Domain.inRotatedQCone()
                )

        # objective
        uv = Expr.add(u, v)
        M.objective(
            ObjectiveSense.Minimize,
            Expr.mul(h/2.0, Expr.dot(np.tile(np.asarray(w), N), Expr.flatten(uv)))
        )

        M.solve()

        rho_sol = np.array(rho.level()).reshape(N+1, n)
        rhobar_sol = np.array(rhobar.level()).reshape(N, n)
        m_sol = np.array(m.level()).reshape(N, a)

    # reconstruct custom warm-start fields
    q = rhobar_sol.copy()
    rho_minus = rhobar_sol[:, src]
    rho_plus  = rhobar_sol[:, dst]
    denom = rho_minus + rho_plus
    vartheta = np.where(denom > 0.0, 2.0 * rho_minus * rho_plus / denom, 0.0)

    return {
        "rho": rho_sol,
        "m": m_sol,
        "rho_bar": rhobar_sol,
        "q": q,
        "rho_minus": rho_minus,
        "rho_plus": rho_plus,
        "vartheta_har": vartheta,
    }
```

---

## 11. Success criterion

The implementation is successful if:

1. the MOSEK model solves without modeling errors,
2. continuity residual is numerically small,
3. endpoint constraints are exactly satisfied,
4. all `rho` and `rhobar` are nonnegative,
5. the returned fields can be consumed directly as warm start by the custom solver.


[1]: https://docs.mosek.com/latest/pythonfusion/tutorial-cqo-shared.html "7.2 Conic Quadratic Optimization — MOSEK Fusion API for Python 11.1.9"