#include "graphot/ceh_linear.hpp"

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

#include "graphot/runtime.hpp"

#if defined(GRAPHOT_USE_OPENMP)
#include <omp.h>
#endif

namespace graphot::core {
namespace {

struct CgResult {
  std::vector<double> x;
  double residual;
  int iters_used;
};

template <typename MatVec, typename Preconditioner>
CgResult conjugate_gradient(MatVec &&matvec, const std::vector<double> &b,
                            int max_iters, double tol,
                            const std::optional<std::vector<double>> &x0,
                            Preconditioner &&preconditioner) {
  std::vector<double> x =
      x0.has_value() ? *x0 : std::vector<double>(b.size(), 0.0);
  std::vector<double> ap;
  matvec(x, ap);

  std::vector<double> r(b.size(), 0.0);
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for if (should_parallelize(b.size()))
#endif
  for (std::size_t idx = 0; idx < b.size(); ++idx) {
    r[idx] = b[idx] - ap[idx];
  }
  std::vector<double> z;
  preconditioner(r, z);
  std::vector<double> p = z;
  double rz_old = vector_dot(r, z);
  double residual = vector_norm(r);
  int iters_used = 0;
  if (residual <= tol) {
    return {std::move(x), residual, iters_used};
  }

  for (int iter = 0; iter < max_iters; ++iter) {
    matvec(p, ap);
    const double denom = std::max(vector_dot(p, ap), kTiny);
    const double alpha = rz_old / denom;
    vector_axpy(x, p, alpha);
    vector_axpy(r, ap, -alpha);
    residual = vector_norm(r);
    ++iters_used;
    if (residual <= tol) {
      break;
    }
    preconditioner(r, z);
    const double rz_new = vector_dot(r, z);
    const double beta = rz_new / std::max(rz_old, kTiny);
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for if (should_parallelize(p.size()))
#endif
    for (std::size_t idx = 0; idx < p.size(); ++idx) {
      p[idx] = z[idx] + beta * p[idx];
    }
    rz_old = rz_new;
  }
  return {std::move(x), residual, iters_used};
}

} // namespace

void continuity_residual(const GraphData &graph, const std::vector<double> &rho,
                         const std::vector<double> &m,
                         const std::vector<double> &rho_a,
                         const std::vector<double> &rho_b, int num_steps,
                         std::vector<double> &out) {
  out.assign(static_cast<std::size_t>(num_steps) *
                 static_cast<std::size_t>(graph.num_nodes),
             0.0);
  const double h = 1.0 / static_cast<double>(num_steps);
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for if (should_parallelize(                               \
                                 static_cast <std::size_t>(num_steps) *        \
                                 (graph.num_nodes + graph.num_edges)))
#endif
  for (int step = 0; step < num_steps; ++step) {
    for (int node = 0; node < graph.num_nodes; ++node) {
      double delta = rho[row_major(step + 1, node, graph.num_nodes)] -
                     rho[row_major(step, node, graph.num_nodes)];
      if (step == 0) {
        delta = rho[row_major(1, node, graph.num_nodes)] - rho_a[node];
      }
      if (step == num_steps - 1) {
        delta =
            rho_b[node] - rho[row_major(num_steps - 1, node, graph.num_nodes)];
      }
      out[row_major(step, node, graph.num_nodes)] = delta / h;
    }
    for (int edge = 0; edge < graph.num_edges; ++edge) {
      const double contribution =
          0.5 * graph.q[edge] *
          (m[row_major(step, graph.rev[edge], graph.num_edges)] -
           m[row_major(step, edge, graph.num_edges)]);
      out[row_major(step, graph.src[edge], graph.num_nodes)] += contribution;
    }
  }
}

void solve_tridiagonal(const std::vector<double> &lower,
                       const std::vector<double> &diag,
                       const std::vector<double> &upper,
                       const std::vector<double> &rhs,
                       std::vector<double> &out) {
  const int size = static_cast<int>(rhs.size());
  std::vector<double> c_prime(size, 0.0);
  std::vector<double> d_prime(size, 0.0);
  c_prime[0] = size > 1 ? upper[0] / diag[0] : 0.0;
  d_prime[0] = rhs[0] / diag[0];
  for (int idx = 1; idx < size; ++idx) {
    const double denom = diag[idx] - lower[idx] * c_prime[idx - 1];
    c_prime[idx] = idx < size - 1 ? upper[idx] / denom : 0.0;
    d_prime[idx] = (rhs[idx] - lower[idx] * d_prime[idx - 1]) / denom;
  }
  out.assign(size, 0.0);
  out[size - 1] = d_prime[size - 1];
  for (int idx = size - 2; idx >= 0; --idx) {
    out[idx] = d_prime[idx] - c_prime[idx] * out[idx + 1];
  }
}

void solve_tridiagonal_javg(const std::vector<double> &rhs, int num_steps,
                            int num_nodes, std::vector<double> &out) {
  std::vector<double> lower(num_steps, 0.25);
  std::vector<double> diag(num_steps, 1.5);
  std::vector<double> upper(num_steps, 0.25);
  lower[0] = 0.0;
  upper[num_steps - 1] = 0.0;
  diag[0] = 1.25;
  diag[num_steps - 1] = 1.25;

  out.assign(rhs.size(), 0.0);
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel if (should_parallelize(                                   \
                             static_cast <std::size_t>(num_nodes) *            \
                             num_steps))
#endif
  {
    std::vector<double> rhs_node(num_steps, 0.0);
    std::vector<double> sol;
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp for schedule(static)
#endif
    for (int node = 0; node < num_nodes; ++node) {
      for (int step = 0; step < num_steps; ++step) {
        rhs_node[step] = rhs[row_major(step, node, num_nodes)];
      }
      solve_tridiagonal(lower, diag, upper, rhs_node, sol);
      for (int step = 0; step < num_steps; ++step) {
        out[row_major(step, node, num_nodes)] = sol[step];
      }
    }
  }
}

void apply_ceh_constraint_zero_boundary(const GraphData &graph,
                                        const std::vector<double> &drho_int,
                                        const std::vector<double> &dm,
                                        int num_steps,
                                        std::vector<double> &out) {
  out.assign(static_cast<std::size_t>(num_steps) *
                 static_cast<std::size_t>(graph.num_nodes),
             0.0);
  const double h = 1.0 / static_cast<double>(num_steps);
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for if (should_parallelize(                               \
                                 static_cast <std::size_t>(num_steps) *        \
                                 (graph.num_nodes + graph.num_edges)))
#endif
  for (int step = 0; step < num_steps; ++step) {
    for (int node = 0; node < graph.num_nodes; ++node) {
      double value = 0.0;
      if (step == 0) {
        value = drho_int[row_major(0, node, graph.num_nodes)] / h;
      } else if (step == num_steps - 1) {
        value = -drho_int[row_major(num_steps - 2, node, graph.num_nodes)] / h;
      } else {
        value = (drho_int[row_major(step, node, graph.num_nodes)] -
                 drho_int[row_major(step - 1, node, graph.num_nodes)]) /
                h;
      }
      out[row_major(step, node, graph.num_nodes)] = value;
    }
    for (int edge = 0; edge < graph.num_edges; ++edge) {
      const double contribution =
          0.5 * graph.q[edge] *
          (dm[row_major(step, graph.rev[edge], graph.num_edges)] -
           dm[row_major(step, edge, graph.num_edges)]);
      out[row_major(step, graph.src[edge], graph.num_nodes)] += contribution;
    }
  }
}

void apply_ceh_constraint_transpose_zero_boundary(
    const GraphData &graph, const std::vector<double> &phi, int num_steps,
    std::vector<double> &drho_int_adj, std::vector<double> &dm_adj) {
  drho_int_adj.assign(static_cast<std::size_t>(num_steps - 1) *
                          static_cast<std::size_t>(graph.num_nodes),
                      0.0);
  dm_adj.assign(static_cast<std::size_t>(num_steps) *
                    static_cast<std::size_t>(graph.num_edges),
                0.0);
  const double h = 1.0 / static_cast<double>(num_steps);
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for if (should_parallelize(drho_int_adj.size()))
#endif
  for (int step = 0; step < num_steps - 1; ++step) {
    for (int node = 0; node < graph.num_nodes; ++node) {
      drho_int_adj[row_major(step, node, graph.num_nodes)] =
          (phi[row_major(step, node, graph.num_nodes)] -
           phi[row_major(step + 1, node, graph.num_nodes)]) /
          h;
    }
  }
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for if (should_parallelize(dm_adj.size()))
#endif
  for (int step = 0; step < num_steps; ++step) {
    for (int edge = 0; edge < graph.num_edges; ++edge) {
      dm_adj[row_major(step, edge, graph.num_edges)] =
          0.5 * (graph.q[graph.rev[edge]] *
                     phi[row_major(step, graph.dst[edge], graph.num_nodes)] -
                 graph.q[edge] *
                     phi[row_major(step, graph.src[edge], graph.num_nodes)]);
    }
  }
}

void apply_jacobi_preconditioner(const GraphData &graph,
                                 const std::vector<double> &value,
                                 int num_steps, std::vector<double> &out) {
  out = value;
  project_zero_mean(out);
  const double h = 1.0 / static_cast<double>(num_steps);
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for if (should_parallelize(out.size()))
#endif
  for (int step = 0; step < num_steps; ++step) {
    const double time_diag =
        (step == 0 || step == num_steps - 1) ? 1.0 / (h * h) : 2.0 / (h * h);
    for (int node = 0; node < graph.num_nodes; ++node) {
      const double pi_inv = 1.0 / std::max(graph.pi[node], kTiny);
      const double diag =
          std::max((time_diag + 0.5 * graph.out_rate[node]) * pi_inv, 1e-12);
      out[row_major(step, node, graph.num_nodes)] /= diag;
    }
  }
  project_zero_mean(out);
}

void apply_block_jacobi_preconditioner(const GraphData &graph,
                                       const std::vector<double> &value,
                                       int num_steps,
                                       std::vector<double> &out) {
  out = value;
  project_zero_mean(out);
  const double h = 1.0 / static_cast<double>(num_steps);
  std::vector<double> lower(num_steps, -1.0 / (h * h));
  std::vector<double> upper(num_steps, -1.0 / (h * h));
  lower[0] = 0.0;
  upper[num_steps - 1] = 0.0;
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel if (should_parallelize(                                   \
                             static_cast <std::size_t>(graph.num_nodes) *      \
                             num_steps))
#endif
  {
    std::vector<double> diag(num_steps, 0.0);
    std::vector<double> rhs_node(num_steps, 0.0);
    std::vector<double> lower_node(num_steps, 0.0);
    std::vector<double> upper_node(num_steps, 0.0);
    std::vector<double> sol;
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp for schedule(static)
#endif
    for (int node = 0; node < graph.num_nodes; ++node) {
      const double pi_inv = 1.0 / std::max(graph.pi[node], kTiny);
      for (int step = 0; step < num_steps; ++step) {
        const double time_diag = (step == 0 || step == num_steps - 1)
                                     ? 1.0 / (h * h)
                                     : 2.0 / (h * h);
        diag[step] =
            std::max((time_diag + 0.5 * graph.out_rate[node]) * pi_inv, 1e-12);
        lower_node[step] = lower[step] * pi_inv;
        upper_node[step] = upper[step] * pi_inv;
        rhs_node[step] = out[row_major(step, node, graph.num_nodes)];
      }
      solve_tridiagonal(lower_node, diag, upper_node, rhs_node, sol);
      for (int step = 0; step < num_steps; ++step) {
        out[row_major(step, node, graph.num_nodes)] = sol[step];
      }
    }
  }
  project_zero_mean(out);
}

void project_ceh(const GraphData &graph, const std::vector<double> &rho,
                 const std::vector<double> &m, const std::vector<double> &rho_a,
                 const std::vector<double> &rho_b, int num_steps,
                 int cg_max_iters, double cg_tol,
                 const std::optional<std::vector<double>> &phi0,
                 bool cg_warm_start, bool block_jacobi,
                 std::vector<double> &rho_pr, std::vector<double> &m_pr,
                 std::vector<double> &phi, CehStats &stats) {
  auto matvec = [&](const std::vector<double> &value,
                    std::vector<double> &out) {
    std::vector<double> phi_local = value;
    project_zero_mean(phi_local);
    std::vector<double> drho_int_adj;
    std::vector<double> dm_adj;
    apply_ceh_constraint_transpose_zero_boundary(graph, phi_local, num_steps,
                                                 drho_int_adj, dm_adj);
    std::vector<double> drho_int(drho_int_adj.size(), 0.0);
    std::vector<double> dm(dm_adj.size(), 0.0);
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for if (should_parallelize(drho_int.size()))
#endif
    for (int step = 0; step < num_steps - 1; ++step) {
      for (int node = 0; node < graph.num_nodes; ++node) {
        drho_int[row_major(step, node, graph.num_nodes)] =
            drho_int_adj[row_major(step, node, graph.num_nodes)] /
            std::max(graph.pi[node], kTiny);
      }
    }
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for if (should_parallelize(dm.size()))
#endif
    for (int step = 0; step < num_steps; ++step) {
      for (int edge = 0; edge < graph.num_edges; ++edge) {
        dm[row_major(step, edge, graph.num_edges)] =
            dm_adj[row_major(step, edge, graph.num_edges)] /
            std::max(graph.edge_weight[edge], kTiny);
      }
    }
    apply_ceh_constraint_zero_boundary(graph, drho_int, dm, num_steps, out);
    project_zero_mean(out);
  };

  std::vector<double> rhs;
  continuity_residual(graph, rho, m, rho_a, rho_b, num_steps, rhs);
  project_zero_mean(rhs);

  auto preconditioner = [&](const std::vector<double> &value,
                            std::vector<double> &out) {
    if (block_jacobi) {
      apply_block_jacobi_preconditioner(graph, value, num_steps, out);
    } else {
      apply_jacobi_preconditioner(graph, value, num_steps, out);
    }
  };

  const std::optional<std::vector<double>> x0 =
      (cg_warm_start && phi0.has_value())
          ? phi0
          : std::optional<std::vector<double>>{};
  CgResult cg =
      conjugate_gradient(matvec, rhs, cg_max_iters, cg_tol, x0, preconditioner);
  phi = std::move(cg.x);
  project_zero_mean(phi);

  std::vector<double> drho_int_adj;
  std::vector<double> dm_adj;
  apply_ceh_constraint_transpose_zero_boundary(graph, phi, num_steps,
                                               drho_int_adj, dm_adj);

  rho_pr = rho;
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for if (should_parallelize(                               \
                                 static_cast <std::size_t>(graph.num_nodes)))
#endif
  for (int node = 0; node < graph.num_nodes; ++node) {
    rho_pr[row_major(0, node, graph.num_nodes)] = rho_a[node];
    rho_pr[row_major(num_steps, node, graph.num_nodes)] = rho_b[node];
  }
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for if (should_parallelize(                               \
                                 static_cast <std::size_t>(num_steps - 1) *    \
                                 graph.num_nodes))
#endif
  for (int step = 1; step < num_steps; ++step) {
    for (int node = 0; node < graph.num_nodes; ++node) {
      rho_pr[row_major(step, node, graph.num_nodes)] =
          rho[row_major(step, node, graph.num_nodes)] -
          drho_int_adj[row_major(step - 1, node, graph.num_nodes)] /
              std::max(graph.pi[node], kTiny);
    }
  }
  m_pr = m;
#if defined(GRAPHOT_USE_OPENMP)
#pragma omp parallel for if (should_parallelize(                               \
                                 static_cast <std::size_t>(num_steps) *        \
                                 graph.num_edges))
#endif
  for (int step = 0; step < num_steps; ++step) {
    for (int edge = 0; edge < graph.num_edges; ++edge) {
      m_pr[row_major(step, edge, graph.num_edges)] =
          m[row_major(step, edge, graph.num_edges)] -
          dm_adj[row_major(step, edge, graph.num_edges)] /
              std::max(graph.edge_weight[edge], kTiny);
    }
  }
  stats = {cg.residual, cg.iters_used};
}

} // namespace graphot::core
