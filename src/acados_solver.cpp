#include "quadrotor/acados_solver.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <dlfcn.h>
#include <stdexcept>
#include <vector>

extern "C" {
#include "acados_c/external_function_interface.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/ocp_qp_interface.h"
#include "acados/ocp_nlp/ocp_nlp_common.h"
#include "acados/ocp_qp/ocp_qp_xcond_solver.h"
#include "quadrotor_generated.h"
}

namespace quad {

std::string toString(SolverKind kind) {
  switch (kind) {
    case SolverKind::Daocp: return "daocp";
    case SolverKind::Hpipm: return "hpipm";
    case SolverKind::Daqp: return "daqp";
    case SolverKind::Qpoases: return "qpoases";
  }
  return "unknown";
}

SolverKind parseSolver(const std::string& name) {
  if (name == "daocp") return SolverKind::Daocp;
  if (name == "hpipm") return SolverKind::Hpipm;
  if (name == "daqp") return SolverKind::Daqp;
  if (name == "qpoases" || name == "qpoases_e") return SolverKind::Qpoases;
  throw std::invalid_argument("unknown solver: " + name);
}

bool solverAvailable(SolverKind kind) {
  const char* symbol = nullptr;
  switch (kind) {
    case SolverKind::Daocp: symbol = "ocp_qp_daocp_config_initialize_default"; break;
    case SolverKind::Hpipm: symbol = "ocp_qp_hpipm_config_initialize_default"; break;
    case SolverKind::Daqp: symbol = "dense_qp_daqp_config_initialize_default"; break;
    case SolverKind::Qpoases: symbol = "dense_qp_qpoases_config_initialize_default"; break;
  }
  return symbol && dlsym(RTLD_DEFAULT, symbol);
}

namespace {

#define DECLARE_ACADOS_CASADI_BRIDGE(BASE) \
  int BASE##_acados_bridge(const double** arg, double** res, int* iw, double* w, void* mem) { \
    return BASE(arg, res, iw, w, static_cast<int>(reinterpret_cast<std::intptr_t>(mem))); \
  }

// CasADi represents its checkout handle as an int. acados stores that handle
// in a void* slot in its generic external-function interface.
DECLARE_ACADOS_CASADI_BRIDGE(quadrotor_expl_vde_forw)
DECLARE_ACADOS_CASADI_BRIDGE(quadrotor_expl_ode_fun)
DECLARE_ACADOS_CASADI_BRIDGE(quadrotor_expl_vde_adj)
DECLARE_ACADOS_CASADI_BRIDGE(quadrotor_cost_y_fun)
DECLARE_ACADOS_CASADI_BRIDGE(quadrotor_cost_y_fun_jac_ut_xt)
DECLARE_ACADOS_CASADI_BRIDGE(quadrotor_cost_y_e_fun)
DECLARE_ACADOS_CASADI_BRIDGE(quadrotor_cost_y_e_fun_jac_ut_xt)

#define MAP_CASADI(FUN, BASE) do { \
  (FUN).casadi_fun = &BASE##_acados_bridge; \
  (FUN).casadi_work = &BASE##_work; \
  (FUN).casadi_sparsity_in = &BASE##_sparsity_in; \
  (FUN).casadi_sparsity_out = &BASE##_sparsity_out; \
  (FUN).casadi_n_in = &BASE##_n_in; \
  (FUN).casadi_n_out = &BASE##_n_out; \
} while (false)

ocp_qp_solver_t qpPlan(SolverKind kind) {
  if (!solverAvailable(kind))
    throw std::runtime_error(toString(kind) + " is not enabled in the installed acados library");
  switch (kind) {
    case SolverKind::Daocp: return PARTIAL_CONDENSING_DAOCP;
    case SolverKind::Hpipm: return PARTIAL_CONDENSING_HPIPM;
    case SolverKind::Daqp: return FULL_CONDENSING_DAQP;
    case SolverKind::Qpoases: return FULL_CONDENSING_QPOASES;
  }
  throw std::invalid_argument("invalid solver kind");
}

std::array<double, 16> pathYref(const FlatReference& r) {
  return {r.state.position.x, r.state.position.y, r.state.position.z,
          0.0, 0.0, 0.0,
          r.state.velocity.x, r.state.velocity.y, r.state.velocity.z,
          r.state.body_rate.x, r.state.body_rate.y, r.state.body_rate.z,
          r.control[0], r.control[1], r.control[2], r.control[3]};
}

std::array<double, 12> terminalYref(const FlatReference& r) {
  return {r.state.position.x, r.state.position.y, r.state.position.z,
          0.0, 0.0, 0.0,
          r.state.velocity.x, r.state.velocity.y, r.state.velocity.z,
          r.state.body_rate.x, r.state.body_rate.y, r.state.body_rate.z};
}

std::array<double, 4> attitudeParameter(const FlatReference& r) {
  return {r.state.attitude.w, r.state.attitude.x, r.state.attitude.y, r.state.attitude.z};
}

}  // namespace

struct AcadosController::Impl {
  ocp_nlp_plan_t* plan{};
  ocp_nlp_config* config{};
  ocp_nlp_dims* dims{};
  ocp_nlp_in* in{};
  ocp_nlp_out* out{};
  void* opts{};
  ocp_nlp_solver* solver{};

  std::array<external_function_param_casadi, kN> expl_vde_forw{};
  std::array<external_function_param_casadi, kN> expl_ode_fun{};
  std::array<external_function_param_casadi, kN> expl_vde_adj{};
  std::array<external_function_param_casadi, kN> cost_y_fun{};
  std::array<external_function_param_casadi, kN> cost_y_jac{};
  external_function_param_casadi cost_y_e_fun{};
  external_function_param_casadi cost_y_e_jac{};
  bool functions_created{false};
  double previous_time_sim{};

  ~Impl() {
    if (solver) ocp_nlp_solver_destroy(solver);
    if (out) ocp_nlp_out_destroy(out);
    if (in) ocp_nlp_in_destroy(in);
    if (opts) ocp_nlp_solver_opts_destroy(opts);
    if (dims) ocp_nlp_dims_destroy(dims);
    if (config) ocp_nlp_config_destroy(config);
    if (plan) ocp_nlp_plan_destroy(plan);
    if (functions_created) {
      for (int i = 0; i < kN; ++i) {
        external_function_param_casadi_free(&expl_vde_forw[static_cast<std::size_t>(i)]);
        external_function_param_casadi_free(&expl_ode_fun[static_cast<std::size_t>(i)]);
        external_function_param_casadi_free(&expl_vde_adj[static_cast<std::size_t>(i)]);
        external_function_param_casadi_free(&cost_y_fun[static_cast<std::size_t>(i)]);
        external_function_param_casadi_free(&cost_y_jac[static_cast<std::size_t>(i)]);
      }
      external_function_param_casadi_free(&cost_y_e_fun);
      external_function_param_casadi_free(&cost_y_e_jac);
    }
  }
};

AcadosController::AcadosController(SolverKind solver_kind)
    : impl_(std::make_unique<Impl>()), solver_kind_(solver_kind) {
  Impl& s = *impl_;
  s.plan = ocp_nlp_plan_create(kN);
  s.plan->nlp_solver = SQP_RTI;
  s.plan->ocp_qp_solver_plan.qp_solver = qpPlan(solver_kind);
  s.plan->regularization = NO_REGULARIZE;
  for (int i = 0; i <= kN; ++i) {
    s.plan->nlp_cost[i] = NONLINEAR_LS;
    s.plan->nlp_constraints[i] = BGH;
    if (i < kN) {
      s.plan->nlp_dynamics[i] = CONTINUOUS_MODEL;
      s.plan->sim_solver_plan[i].sim_solver = ERK;
    }
  }
  s.config = ocp_nlp_config_create(*s.plan);
  s.dims = ocp_nlp_dims_create(s.config);

  std::array<int, kN+1> nx{}, nu{}, nz{}, ns{}, np{}, ny{};
  std::array<int, kN+1> nbx{}, nbu{}, ng{}, nh{}, nbxe{};
  for (int i = 0; i <= kN; ++i) {
    nx[static_cast<std::size_t>(i)] = kNx;
    nu[static_cast<std::size_t>(i)] = i < kN ? kNu : 0;
    np[static_cast<std::size_t>(i)] = 4;
    ny[static_cast<std::size_t>(i)] = i < kN ? 16 : 12;
    nbx[static_cast<std::size_t>(i)] = i == 0 ? kNx : 3;
    nbu[static_cast<std::size_t>(i)] = i < kN ? kNu : 0;
    nbxe[static_cast<std::size_t>(i)] = i == 0 ? kNx : 0;
  }
  ocp_nlp_dims_set_opt_vars(s.config, s.dims, "nx", nx.data());
  ocp_nlp_dims_set_opt_vars(s.config, s.dims, "nu", nu.data());
  ocp_nlp_dims_set_opt_vars(s.config, s.dims, "nz", nz.data());
  ocp_nlp_dims_set_opt_vars(s.config, s.dims, "ns", ns.data());
  ocp_nlp_dims_set_opt_vars(s.config, s.dims, "np", np.data());
  for (int i = 0; i <= kN; ++i) {
    ocp_nlp_dims_set_cost(s.config, s.dims, i, "ny", &ny[static_cast<std::size_t>(i)]);
    ocp_nlp_dims_set_constraints(s.config, s.dims, i, "nbx", &nbx[static_cast<std::size_t>(i)]);
    ocp_nlp_dims_set_constraints(s.config, s.dims, i, "nbu", &nbu[static_cast<std::size_t>(i)]);
    ocp_nlp_dims_set_constraints(s.config, s.dims, i, "ng", &ng[static_cast<std::size_t>(i)]);
    ocp_nlp_dims_set_constraints(s.config, s.dims, i, "nh", &nh[static_cast<std::size_t>(i)]);
    ocp_nlp_dims_set_constraints(s.config, s.dims, i, "nbxe", &nbxe[static_cast<std::size_t>(i)]);
  }

  external_function_opts fun_opts;
  external_function_opts_set_to_default(&fun_opts);
  fun_opts.external_workspace = false;
  for (int i = 0; i < kN; ++i) {
    const std::size_t j = static_cast<std::size_t>(i);
    MAP_CASADI(s.expl_vde_forw[j], quadrotor_expl_vde_forw);
    MAP_CASADI(s.expl_ode_fun[j], quadrotor_expl_ode_fun);
    MAP_CASADI(s.expl_vde_adj[j], quadrotor_expl_vde_adj);
    MAP_CASADI(s.cost_y_fun[j], quadrotor_cost_y_fun);
    MAP_CASADI(s.cost_y_jac[j], quadrotor_cost_y_fun_jac_ut_xt);
    external_function_param_casadi_create(&s.expl_vde_forw[j], 4, &fun_opts);
    external_function_param_casadi_create(&s.expl_ode_fun[j], 4, &fun_opts);
    external_function_param_casadi_create(&s.expl_vde_adj[j], 4, &fun_opts);
    external_function_param_casadi_create(&s.cost_y_fun[j], 4, &fun_opts);
    external_function_param_casadi_create(&s.cost_y_jac[j], 4, &fun_opts);
  }
  MAP_CASADI(s.cost_y_e_fun, quadrotor_cost_y_e_fun);
  MAP_CASADI(s.cost_y_e_jac, quadrotor_cost_y_e_fun_jac_ut_xt);
  external_function_param_casadi_create(&s.cost_y_e_fun, 4, &fun_opts);
  external_function_param_casadi_create(&s.cost_y_e_jac, 4, &fun_opts);
  s.functions_created = true;

  s.out = ocp_nlp_out_create(s.config, s.dims);
  s.in = ocp_nlp_in_create(s.config, s.dims);

  std::array<double, 16*16> w{};
  const std::array<double, 16> weights{
    200.0, 200.0, 500.0, 5.0, 5.0, 200.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 6.0, 6.0, 6.0, 6.0};
  for (int i = 0; i < 16; ++i) w[static_cast<std::size_t>(i + 16*i)] = weights[static_cast<std::size_t>(i)];
  std::array<double, 12*12> w_e{};
  for (int i = 0; i < 12; ++i) w_e[static_cast<std::size_t>(i + 12*i)] = weights[static_cast<std::size_t>(i)];

  std::array<double, 4> zero_param{1.0, 0.0, 0.0, 0.0};
  for (int i = 0; i < kN; ++i) {
    const std::size_t j = static_cast<std::size_t>(i);
    double dt = kHorizonDt;
    ocp_nlp_in_set(s.config, s.dims, s.in, i, "Ts", &dt);
    ocp_nlp_cost_model_set(s.config, s.dims, s.in, i, "W", w.data());
    ocp_nlp_cost_model_set(s.config, s.dims, s.in, i, "nls_y_fun", &s.cost_y_fun[j]);
    ocp_nlp_cost_model_set(s.config, s.dims, s.in, i, "nls_y_fun_jac", &s.cost_y_jac[j]);
    ocp_nlp_dynamics_model_set(s.config, s.dims, s.in, i, "expl_vde_forw", &s.expl_vde_forw[j]);
    ocp_nlp_dynamics_model_set(s.config, s.dims, s.in, i, "expl_ode_fun", &s.expl_ode_fun[j]);
    ocp_nlp_dynamics_model_set(s.config, s.dims, s.in, i, "expl_vde_adj", &s.expl_vde_adj[j]);
    s.expl_vde_forw[j].set_param(&s.expl_vde_forw[j], zero_param.data());
    s.expl_ode_fun[j].set_param(&s.expl_ode_fun[j], zero_param.data());
    s.expl_vde_adj[j].set_param(&s.expl_vde_adj[j], zero_param.data());
  }
  ocp_nlp_cost_model_set(s.config, s.dims, s.in, kN, "W", w_e.data());
  ocp_nlp_cost_model_set(s.config, s.dims, s.in, kN, "nls_y_fun", &s.cost_y_e_fun);
  ocp_nlp_cost_model_set(s.config, s.dims, s.in, kN, "nls_y_fun_jac", &s.cost_y_e_jac);

  const Parameters p;
  std::array<int, kNu> idxbu{0, 1, 2, 3};
  std::array<double, kNu> lbu{p.thrust_min, p.thrust_min, p.thrust_min, p.thrust_min};
  std::array<double, kNu> ubu{p.thrust_max, p.thrust_max, p.thrust_max, p.thrust_max};
  for (int i = 0; i < kN; ++i) {
    ocp_nlp_constraints_model_set(s.config, s.dims, s.in, s.out, i, "idxbu", idxbu.data());
    ocp_nlp_constraints_model_set(s.config, s.dims, s.in, s.out, i, "lbu", lbu.data());
    ocp_nlp_constraints_model_set(s.config, s.dims, s.in, s.out, i, "ubu", ubu.data());
  }
  std::array<int, kNx> idxbx0{};
  std::array<int, kNx> idxbxe0{};
  for (int i = 0; i < kNx; ++i) idxbx0[static_cast<std::size_t>(i)] = idxbxe0[static_cast<std::size_t>(i)] = i;
  State default_state;
  default_state.position.z = 5.0;
  const auto x_default = toArray(default_state);
  ocp_nlp_constraints_model_set(s.config, s.dims, s.in, s.out, 0, "idxbx", idxbx0.data());
  ocp_nlp_constraints_model_set(s.config, s.dims, s.in, s.out, 0, "lbx", const_cast<double*>(x_default.data()));
  ocp_nlp_constraints_model_set(s.config, s.dims, s.in, s.out, 0, "ubx", const_cast<double*>(x_default.data()));
  // Explicit equality markers are required by DAOCP; harmless for the others.
  ocp_nlp_constraints_model_set(s.config, s.dims, s.in, s.out, 0, "idxbxe", idxbxe0.data());
  std::array<int, 3> idxbx_rate{10, 11, 12};
  std::array<double, 3> lb_rate{-10.0, -10.0, -4.0};
  std::array<double, 3> ub_rate{10.0, 10.0, 4.0};
  for (int i = 1; i <= kN; ++i) {
    ocp_nlp_constraints_model_set(s.config, s.dims, s.in, s.out, i, "idxbx", idxbx_rate.data());
    ocp_nlp_constraints_model_set(s.config, s.dims, s.in, s.out, i, "lbx", lb_rate.data());
    ocp_nlp_constraints_model_set(s.config, s.dims, s.in, s.out, i, "ubx", ub_rate.data());
  }

  s.opts = ocp_nlp_solver_opts_create(s.config, s.dims);
  int integrator_stages = 4;
  int integrator_steps = 1;
  for (int i = 0; i < kN; ++i) {
    ocp_nlp_solver_opts_set_at_stage(s.config, s.opts, i, "dynamics_num_stages", &integrator_stages);
    ocp_nlp_solver_opts_set_at_stage(s.config, s.opts, i, "dynamics_num_steps", &integrator_steps);
  }
  int qp_iter_max = 1000;
  int qp_warm_start = 1;
  int print_level = 0;
  double qp_tol = solver_kind == SolverKind::Hpipm ? 1e-3 : 1e-8;
  ocp_nlp_solver_opts_set(s.config, s.opts, "qp_iter_max", &qp_iter_max);
  ocp_nlp_solver_opts_set(s.config, s.opts, "qp_warm_start", &qp_warm_start);
  ocp_nlp_solver_opts_set(s.config, s.opts, "print_level", &print_level);
  // Preserve the QP solver's working set after the first call, but keep the
  // first measured call a genuine cold-working-set solve.
  bool warm_first = false;
  bool warm_first_from_nlp = false;
  ocp_nlp_solver_opts_set(s.config, s.opts, "warm_start_first_qp", &warm_first);
  ocp_nlp_solver_opts_set(s.config, s.opts, "warm_start_first_qp_from_nlp", &warm_first_from_nlp);
  if (solver_kind == SolverKind::Daocp || solver_kind == SolverKind::Hpipm) {
    int cond_n = kN;
    ocp_nlp_solver_opts_set(s.config, s.opts, "qp_cond_N", &cond_n);
  }
  if (solver_kind == SolverKind::Daocp) {
    int most_violated = 1;
    ocp_nlp_solver_opts_set(s.config, s.opts, "qp_selection_strategy", &most_violated);
  } else if (solver_kind == SolverKind::Hpipm) {
    ocp_nlp_solver_opts_set(s.config, s.opts, "qp_hpipm_mode", const_cast<char*>("SPEED_ABS"));
  }
  // HPIPM mode resets its defaults, so common tolerances must be applied last.
  ocp_nlp_solver_opts_set(s.config, s.opts, "qp_tol_stat", &qp_tol);
  ocp_nlp_solver_opts_set(s.config, s.opts, "qp_tol_eq", &qp_tol);
  ocp_nlp_solver_opts_set(s.config, s.opts, "qp_tol_ineq", &qp_tol);
  ocp_nlp_solver_opts_set(s.config, s.opts, "qp_tol_comp", &qp_tol);

  s.solver = ocp_nlp_solver_create(s.config, s.dims, s.opts, s.in);
  const int status = ocp_nlp_precompute(s.solver, s.in, s.out);
  if (status != ACADOS_SUCCESS) throw std::runtime_error("acados precompute failed for " + toString(solver_kind));
}

AcadosController::~AcadosController() = default;

void AcadosController::updateInitialState(const State& state) {
  Impl& s = *impl_;
  const auto x = toArray(state);
  ocp_nlp_constraints_model_set(s.config, s.dims, s.in, s.out, 0, "lbx", const_cast<double*>(x.data()));
  ocp_nlp_constraints_model_set(s.config, s.dims, s.in, s.out, 0, "ubx", const_cast<double*>(x.data()));
}

void AcadosController::updateReferences(const std::vector<FlatReference>& horizon) {
  if (horizon.size() != static_cast<std::size_t>(kN+1)) throw std::invalid_argument("reference horizon must have N+1 samples");
  Impl& s = *impl_;
  for (int i = 0; i < kN; ++i) {
    const std::size_t j = static_cast<std::size_t>(i);
    auto yref = pathYref(horizon[j]);
    auto qref = attitudeParameter(horizon[j]);
    ocp_nlp_cost_model_set(s.config, s.dims, s.in, i, "yref", yref.data());
    s.cost_y_fun[j].set_param(&s.cost_y_fun[j], qref.data());
    s.cost_y_jac[j].set_param(&s.cost_y_jac[j], qref.data());
  }
  auto yref_e = terminalYref(horizon.back());
  auto qref_e = attitudeParameter(horizon.back());
  ocp_nlp_cost_model_set(s.config, s.dims, s.in, kN, "yref", yref_e.data());
  s.cost_y_e_fun.set_param(&s.cost_y_e_fun, qref_e.data());
  s.cost_y_e_jac.set_param(&s.cost_y_e_jac, qref_e.data());
}

void AcadosController::initialize(const std::vector<FlatReference>& horizon) {
  updateReferences(horizon);
  updateInitialState(horizon.front().state);
  Impl& s = *impl_;
  const Parameters parameters;
  for (int i = 0; i <= kN; ++i) {
    auto x = toArray(horizon[static_cast<std::size_t>(i)].state);
    x[10] = std::clamp(x[10], -10.0, 10.0);
    x[11] = std::clamp(x[11], -10.0, 10.0);
    x[12] = std::clamp(x[12], -4.0, 4.0);
    ocp_nlp_out_set(s.config, s.dims, s.out, s.in, i, "x", x.data());
    if (i < kN) {
      auto u = horizon[static_cast<std::size_t>(i)].control;
      for (double& value : u)
        value = std::clamp(value, parameters.thrust_min, parameters.thrust_max);
      ocp_nlp_out_set(s.config, s.dims, s.out, s.in, i, "u", u.data());
    }
  }
}

SolveStats AcadosController::solve() {
  Impl& s = *impl_;
  SolveStats stats;
  const auto start = std::chrono::steady_clock::now();
  stats.status = ocp_nlp_solve(s.solver, s.in, s.out);
  stats.wall_time = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  // The dense DAQP and qpOASES backends do not expose a separate `status`
  // memory field (their acados adapters terminate on an unsupported query).
  // SQP-RTI propagates their QP failure as its return status.
  stats.qp_status = stats.status;
  if (solver_kind_ == SolverKind::Daocp || solver_kind_ == SolverKind::Hpipm) {
    ocp_nlp_get(s.solver, "qp_status", &stats.qp_status);
  }
  ocp_nlp_get(s.solver, "sqp_iter", &stats.sqp_iter);
  ocp_nlp_get(s.solver, "qp_iter", &stats.qp_iter);
  ocp_nlp_get(s.solver, "time_tot", &stats.time_tot);
  ocp_nlp_get(s.solver, "time_lin", &stats.time_lin);
  // SQP-RTI resets its main sub-timers on every call, but this acados version
  // accumulates time_sim. Convert it back to a per-call measurement so the CSV
  // and summaries remain meaningful.
  double cumulative_time_sim{};
  ocp_nlp_get(s.solver, "time_sim", &cumulative_time_sim);
  stats.time_sim = cumulative_time_sim >= s.previous_time_sim
                     ? cumulative_time_sim - s.previous_time_sim
                     : cumulative_time_sim;
  s.previous_time_sim = cumulative_time_sim;
  ocp_nlp_get(s.solver, "time_qp", &stats.time_qp);
  ocp_nlp_get(s.solver, "time_qp_solver_call", &stats.time_qp_solver_call);
  ocp_nlp_get(s.solver, "time_qp_xcond", &stats.time_qp_xcond);

  // Diagnostics are evaluated after reading acados' timers, so they do not
  // contaminate the reported controller solve time.
  ocp_qp_in* qp_in{};
  ocp_qp_out* qp_out{};
  ocp_nlp_get(s.solver, "qp_in", &qp_in);
  ocp_nlp_get(s.solver, "qp_out", &qp_out);
  ocp_qp_inf_norm_residuals(s.dims->qp_solver->orig_dims, qp_in, qp_out,
                            stats.qp_residuals.data());
  ocp_nlp_eval_residuals(s.solver, s.in, s.out);
  ocp_nlp_eval_cost(s.solver, s.in, s.out);
  ocp_nlp_get(s.solver, "res_stat", &stats.nlp_residuals[0]);
  ocp_nlp_get(s.solver, "res_eq", &stats.nlp_residuals[1]);
  ocp_nlp_get(s.solver, "res_ineq", &stats.nlp_residuals[2]);
  ocp_nlp_get(s.solver, "res_comp", &stats.nlp_residuals[3]);
  ocp_nlp_get(s.solver, "cost_value", &stats.objective);
  const Parameters parameters;
  std::array<double, kNx> predicted_x{};
  Control predicted_u{};
  for (int i = 0; i <= kN; ++i) {
    ocp_nlp_out_get(s.config, s.dims, s.out, i, "x", predicted_x.data());
    stats.max_bound_violation = std::max(stats.max_bound_violation,
      std::max({std::abs(predicted_x[10])-10.0,
                std::abs(predicted_x[11])-10.0,
                std::abs(predicted_x[12])-4.0, 0.0}));
    if (i < kN) {
      ocp_nlp_out_get(s.config, s.dims, s.out, i, "u", predicted_u.data());
      for (double value : predicted_u) {
        stats.max_bound_violation = std::max(stats.max_bound_violation,
          std::max({parameters.thrust_min-value, value-parameters.thrust_max, 0.0}));
      }
    }
  }
  return stats;
}

Control AcadosController::control() const {
  Control u{};
  ocp_nlp_out_get(impl_->config, impl_->dims, impl_->out, 0, "u", u.data());
  return u;
}

void AcadosController::shiftWarmStart(const FlatReference& next_terminal_reference) {
  Impl& s = *impl_;
  std::array<std::array<double, kNx>, kN+1> old_x{};
  std::array<Control, kN> old_u{};
  for (int i = 0; i <= kN; ++i) {
    ocp_nlp_out_get(s.config, s.dims, s.out, i, "x", old_x[static_cast<std::size_t>(i)].data());
    if (i < kN) {
      ocp_nlp_out_get(s.config, s.dims, s.out, i, "u", old_u[static_cast<std::size_t>(i)].data());
    }
  }

  // The controller is called at 100 Hz while the shooting grid is 20 Hz.
  // Fractionally shift the primal trajectory by 10/50 of one shooting
  // interval instead of incorrectly advancing it by a complete stage.
  constexpr double alpha = kControlDt/kHorizonDt;
  for (int i = 0; i < kN; ++i) {
    std::array<double, kNx> x{};
    const auto& left = old_x[static_cast<std::size_t>(i)];
    const auto& right = old_x[static_cast<std::size_t>(i+1)];
    for (int j = 0; j < kNx; ++j) {
      x[static_cast<std::size_t>(j)] = (1.0-alpha)*left[static_cast<std::size_t>(j)]
                                     + alpha*right[static_cast<std::size_t>(j)];
    }
    Quaternion q_left{left[3], left[4], left[5], left[6]};
    Quaternion q_right{right[3], right[4], right[5], right[6]};
    if (dot(q_left, q_right) < 0.0) q_right = -q_right;
    const Quaternion q = normalized(q_left*(1.0-alpha) + q_right*alpha);
    x[3] = q.w; x[4] = q.x; x[5] = q.y; x[6] = q.z;
    ocp_nlp_out_set(s.config, s.dims, s.out, s.in, i, "x", x.data());
    if (i < kN-1) {
      Control u{};
      for (int j = 0; j < kNu; ++j) {
        u[static_cast<std::size_t>(j)] =
          (1.0-alpha)*old_u[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]
          + alpha*old_u[static_cast<std::size_t>(i+1)][static_cast<std::size_t>(j)];
      }
      ocp_nlp_out_set(s.config, s.dims, s.out, s.in, i, "u", u.data());
    }
  }
  auto terminal_x = toArray(next_terminal_reference.state);
  ocp_nlp_out_set(s.config, s.dims, s.out, s.in, kN, "x", terminal_x.data());
  Control u = next_terminal_reference.control;
  ocp_nlp_out_set(s.config, s.dims, s.out, s.in, kN-1, "u", u.data());
}

#undef MAP_CASADI
#undef DECLARE_ACADOS_CASADI_BRIDGE

}  // namespace quad
