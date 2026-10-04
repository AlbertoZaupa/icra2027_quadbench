#include "quadrotor/acados_solver.hpp"
#include "quadrotor/timing.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <sched.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace quad {
namespace {

#ifdef QUADROTOR_GRID_BENCHMARK
constexpr bool kGridBenchmark = true;
#else
constexpr bool kGridBenchmark = false;
#endif

struct Options {
  std::vector<SolverKind> solvers{SolverKind::Daocp, SolverKind::Hpipm,
                                  SolverKind::Daqp, SolverKind::Qpoases};
  std::vector<TrajectoryConfig> trajectories{
    trajectoryConfig(TrajectoryKind::Challenging)};
  int repetitions{20};
  double laps{5.0};
  fs::path output{};
  bool warmup{true};
  bool plots{true};
  std::string cpu_affinity{};
};

struct FirstSolve {
  Control control{};
  double objective{};
  std::array<double, 4> qp_res{};
  bool valid{false};
};

struct Aggregate {
  SolverKind solver{};
  std::string trajectory_label{};
  std::vector<double> time_tot;
  std::vector<double> wall;
  std::vector<double> time_lin;
  std::vector<double> time_sim;
  std::vector<double> time_qp;
  std::vector<double> time_qp_solver;
  std::vector<double> time_qp_xcond;
  std::vector<int> sampling_index;
  double position_error_sq{};
  double attitude_error_sq{};
  double velocity_error_sq{};
  double max_position_error{};
  double max_attitude_error{};
  double max_bound_violation{};
  double max_qp_residual{};
  std::uint64_t samples{};
  std::uint64_t failures{};
  std::uint64_t crashes{};
  std::uint64_t lower_active{};
  std::uint64_t upper_active{};
  std::uint64_t rate_active{};
  std::uint64_t qp_iterations{};
  int completed_runs{};
  bool valid{true};
};

std::string timestamp() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  localtime_r(&t, &tm);
  std::ostringstream s;
  s << std::put_time(&tm, "%Y%m%d-%H%M%S");
  return s.str();
}

std::vector<std::string> split(const std::string& text, char delimiter) {
  std::vector<std::string> result;
  std::stringstream stream(text);
  std::string item;
  if (text.empty() || text.back() == delimiter) throw std::invalid_argument("empty list entry");
  while (std::getline(stream, item, delimiter)) {
    if (item.empty()) throw std::invalid_argument("empty list entry");
    result.push_back(item);
  }
  return result;
}

int parseInteger(const std::string& text) {
  std::size_t end = 0;
  const int value = std::stoi(text, &end);
  if (end != text.size()) throw std::invalid_argument("invalid integer: " + text);
  return value;
}

double parseNumber(const std::string& text) {
  std::size_t end = 0;
  const double value = std::stod(text, &end);
  if (end != text.size()) throw std::invalid_argument("invalid number: " + text);
  return value;
}

Options parseOptions(int argc, char** argv) {
  Options o;
  if (kGridBenchmark) { o.trajectories = paperTrajectoryGrid(); o.plots = false; }
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto value = [&]() -> std::string {
      if (i+1 >= argc) throw std::invalid_argument("missing value for " + arg);
      return argv[++i];
    };
    if (arg == "--solver") {
      const std::string v = value();
      o.solvers.clear();
      if (v == "all") o.solvers = {SolverKind::Daocp, SolverKind::Hpipm, SolverKind::Daqp, SolverKind::Qpoases};
      else for (const auto& item : split(v, ',')) o.solvers.push_back(parseSolver(item));
    } else if (arg == "--trajectory" && !kGridBenchmark) {
      const std::string v = value();
      if (v != "challenging" && v != "daocp-worst")
        throw std::invalid_argument("trajectory must be challenging or daocp-worst");
      o.trajectories = {parseTrajectory(v)};
    } else if (arg == "--repetitions") {
      o.repetitions = parseInteger(value());
    } else if (arg == "--laps") {
      o.laps = parseNumber(value());
    } else if (arg == "--output" && !kGridBenchmark) {
      o.output = value();
    } else if (arg == "--cpu-affinity") {
      o.cpu_affinity = value();
    } else if (arg == "--no-warmup") {
      o.warmup = false;
    } else if (arg == "--no-plots" && !kGridBenchmark) {
      o.plots = false;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << (kGridBenchmark ? "quadrotor_grid_benchmark" : "quadrotor_benchmark") << " [options]\n"
        "  --solver all|daocp,hpipm,daqp,qpoases\n"
        "  --repetitions N        measured fresh simulations (default 20)\n"
        "  --laps N               laps per simulation (default 5; fractions allowed)\n"
        "  --cpu-affinity LIST    Linux CPU list, e.g. 3 or 2-3\n"
        "  --no-warmup            skip the unmeasured warm-up simulation\n";
      if (!kGridBenchmark) std::cout <<
        "  --trajectory challenging|daocp-worst (default challenging)\n"
        "  --output DIR           artifact directory (default results/TIMESTAMP)\n"
        "  --no-plots             skip plot generation\n";
      else std::cout << "Runs all 144 Table III trajectories; writes no files.\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  if (o.solvers.empty() || o.trajectories.empty()) throw std::invalid_argument("selection cannot be empty");
  if (o.repetitions <= 0 || !std::isfinite(o.laps) || !(o.laps > 0.0)) throw std::invalid_argument("repetitions and laps must be positive");
  std::set<SolverKind> unique(o.solvers.begin(), o.solvers.end());
  if (unique.size() != o.solvers.size()) throw std::invalid_argument("duplicate solver selection");
  if (!kGridBenchmark && o.output.empty()) o.output = fs::path("results") / timestamp();
  return o;
}

std::set<int> parseCpuList(const std::string& text) {
  std::set<int> cpus;
  for (const std::string& part : split(text, ',')) {
    const auto dash = part.find('-');
    if (dash == std::string::npos) {
      cpus.insert(parseInteger(part));
    } else {
      const int first = parseInteger(part.substr(0, dash));
      const int last = parseInteger(part.substr(dash+1));
      if (first < 0 || last >= CPU_SETSIZE || first > last) throw std::invalid_argument("invalid CPU range: " + part);
      for (int cpu = first; cpu <= last; ++cpu) cpus.insert(cpu);
    }
  }
  return cpus;
}

void applyAffinity(const std::string& requested) {
  if (requested.empty()) return;
  cpu_set_t set;
  CPU_ZERO(&set);
  for (int cpu : parseCpuList(requested)) {
    if (cpu < 0 || cpu >= CPU_SETSIZE) throw std::invalid_argument("CPU index out of range");
    CPU_SET(cpu, &set);
  }
  if (sched_setaffinity(0, sizeof(set), &set) != 0) {
    throw std::runtime_error("sched_setaffinity failed: " + std::string(std::strerror(errno)));
  }
}

std::string currentAffinity() {
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) != 0) return "unavailable";
  std::ostringstream out;
  bool first = true;
  for (int i = 0; i < CPU_SETSIZE; ++i) if (CPU_ISSET(i, &set)) {
    if (!first) out << ',';
    out << i;
    first = false;
  }
  return out.str();
}

std::string cpuModel() {
  std::ifstream in("/proc/cpuinfo");
  std::string line;
  while (std::getline(in, line)) {
    const std::string key = "model name";
    if (line.rfind(key, 0) == 0) {
      const auto colon = line.find(':');
      if (colon != std::string::npos) return line.substr(colon+2);
    }
  }
  return "unknown";
}

std::string jsonEscape(const std::string& value) {
  std::ostringstream out;
  for (const unsigned char c : value) {
    if (c == '"') out << "\\\"";
    else if (c == '\\') out << "\\\\";
    else if (c == '\n') out << "\\n";
    else if (c < 0x20) out << "?";
    else out << c;
  }
  return out.str();
}

void writeMetadata(const Options& options, const fs::path& file) {
  struct utsname uts{};
  uname(&uts);
  std::ofstream out(file);
  out << std::setprecision(17);
  out << "{\n"
      << "  \"schema_version\": 1,\n"
      << "  \"generated_at\": \"" << timestamp() << "\",\n"
      << "  \"paper\": \"https://arxiv.org/abs/2109.01365\",\n"
      << "  \"python\": \"" << jsonEscape(BENCH_PYTHON_EXECUTABLE) << "\",\n"
      << "  \"compiler\": \"" << BENCH_COMPILER_ID << ' ' << BENCH_COMPILER_VERSION << "\",\n"
      << "  \"build_type\": \"" << BENCH_BUILD_TYPE << "\",\n"
      << "  \"cpu\": \"" << jsonEscape(cpuModel()) << "\",\n"
      << "  \"kernel\": \"" << uts.sysname << ' ' << uts.release << ' ' << uts.machine << "\",\n"
      << "  \"affinity\": \"" << currentAffinity() << "\",\n"
      << "  \"configuration\": {\n"
      << "    \"N\": 20, \"horizon_s\": 1.0, \"shooting_interval_s\": 0.05,\n"
      << "    \"controller_interval_s\": 0.01, \"plant_rk4_max_step_s\": 0.002,\n"
      << "    \"repetitions\": " << options.repetitions << ", \"laps\": " << options.laps << ",\n"
      << "    \"trajectory_count\": " << options.trajectories.size() << ",\n"
      << "    \"warmup_simulations\": " << (options.warmup ? 1 : 0) << ",\n"
      << "    \"daocp_selection\": \"MOST_VIOLATED\",\n"
      << "    \"hpipm_mode\": \"SPEED_ABS\",\n"
      << "    \"hpipm_stationarity_residual_available\": false,\n"
      << "    \"qp_iteration_cap\": 1000,\n"
      << "    \"qp_tolerance\": {\"hpipm\": 1e-3, \"other_solvers\": 1e-8},\n"
      << "    \"timing_scope\": \"acados time_tot excludes reference update, diagnostics, serialization, plotting, and plant propagation\",\n"
      << "    \"timing_aggregation\": \"average each sampling index across repetitions\",\n"
      << "    \"timing_plot_statistics\": [\"average\", \"max_per_sampling_average\"]\n"
      << "  }\n} \n";
}

std::vector<FlatReference> makeHorizon(const TrajectoryConfig& trajectory, double time) {
  std::vector<FlatReference> horizon;
  horizon.reserve(kN+1);
  for (int i = 0; i <= kN; ++i) horizon.push_back(referenceAt(trajectory, time+i*kHorizonDt));
  return horizon;
}

bool allFinite(const SolveStats& stats) {
  const std::array<double, 12> values{
    stats.time_tot, stats.time_lin, stats.time_sim, stats.time_qp,
    stats.time_qp_solver_call, stats.time_qp_xcond, stats.wall_time, stats.objective,
    stats.nlp_residuals[0], stats.nlp_residuals[1], stats.nlp_residuals[2], stats.nlp_residuals[3]};
  if (!std::all_of(values.begin(), values.end(), [](double x) { return std::isfinite(x); })) return false;
  return std::all_of(stats.qp_residuals.begin(), stats.qp_residuals.end(),
                     [](double x) { return std::isfinite(x); });
}

double maxQpResidual(const std::array<double, 4>& residuals, SolverKind solver_kind) {
  double maximum = 0.0;
  // SPEED_ABS does not compute the equality-constraint dual, so its
  // stationarity residual (index 0) is not a meaningful KKT diagnostic.
  // Primal feasibility and complementarity remain available.
  const std::size_t first = solver_kind == SolverKind::Hpipm ? 1 : 0;
  for (std::size_t i = first; i < residuals.size(); ++i)
    maximum = std::max(maximum, std::abs(residuals[i]));
  return maximum;
}

double maxQpResidual(const SolveStats& stats, SolverKind solver_kind) {
  return maxQpResidual(stats.qp_residuals, solver_kind);
}

void writeCsvHeader(std::ostream& out) {
  out << "solver,trajectory,repetition,step,time_s,status,qp_status,sqp_iter,qp_iter,"
      << "time_tot_us,time_lin_us,time_sim_us,time_qp_us,time_qp_solver_call_us,time_qp_xcond_us,wall_us,"
      << "objective,nlp_res_stat,nlp_res_eq,nlp_res_ineq,nlp_res_comp,"
      << "qp_res_stat,qp_res_eq,qp_res_ineq,qp_res_comp,max_bound_violation,"
      << "position_error,attitude_error_rad,velocity_error,"
      << "px,py,pz,qw,qx,qy,qz,vx,vy,vz,wx,wy,wz,"
      << "ref_px,ref_py,ref_pz,ref_qw,ref_qx,ref_qy,ref_qz,ref_vx,ref_vy,ref_vz,ref_wx,ref_wy,ref_wz,"
      << "u0,u1,u2,u3,ref_u0,ref_u1,ref_u2,ref_u3\n";
}

void writeStep(std::ostream& out, SolverKind solver, const TrajectoryConfig& trajectory,
               int repetition, int step, double time, const SolveStats& stats,
               const State& state, const FlatReference& ref, const Control& u,
               double position_error, double attitude_error, double velocity_error) {
  out << std::setprecision(17)
      << toString(solver) << ',' << toString(trajectory) << ',' << repetition << ',' << step << ',' << time << ','
      << stats.status << ',' << stats.qp_status << ',' << stats.sqp_iter << ',' << stats.qp_iter << ','
      << 1e6*stats.time_tot << ',' << 1e6*stats.time_lin << ',' << 1e6*stats.time_sim << ','
      << 1e6*stats.time_qp << ',' << 1e6*stats.time_qp_solver_call << ',' << 1e6*stats.time_qp_xcond << ','
      << 1e6*stats.wall_time << ',' << stats.objective << ','
      << stats.nlp_residuals[0] << ',' << stats.nlp_residuals[1] << ',' << stats.nlp_residuals[2] << ',' << stats.nlp_residuals[3] << ','
      << stats.qp_residuals[0] << ',' << stats.qp_residuals[1] << ',' << stats.qp_residuals[2] << ',' << stats.qp_residuals[3] << ','
      << stats.max_bound_violation << ',' << position_error << ',' << attitude_error << ',' << velocity_error << ','
      << state.position.x << ',' << state.position.y << ',' << state.position.z << ','
      << state.attitude.w << ',' << state.attitude.x << ',' << state.attitude.y << ',' << state.attitude.z << ','
      << state.velocity.x << ',' << state.velocity.y << ',' << state.velocity.z << ','
      << state.body_rate.x << ',' << state.body_rate.y << ',' << state.body_rate.z << ','
      << ref.state.position.x << ',' << ref.state.position.y << ',' << ref.state.position.z << ','
      << ref.state.attitude.w << ',' << ref.state.attitude.x << ',' << ref.state.attitude.y << ',' << ref.state.attitude.z << ','
      << ref.state.velocity.x << ',' << ref.state.velocity.y << ',' << ref.state.velocity.z << ','
      << ref.state.body_rate.x << ',' << ref.state.body_rate.y << ',' << ref.state.body_rate.z;
  for (double value : u) out << ',' << value;
  for (double value : ref.control) out << ',' << value;
  out << '\n';
}

bool crashCondition(const State& state, const TrajectoryConfig& c) {
  const double y_radius = c.plane == TrajectoryPlane::Horizontal ? c.rmin : 0.0;
  const double z_radius = c.plane == TrajectoryPlane::Vertical ? c.rmin : 0.0;
  return !finite(state) || state.position.x < -c.rmax-5.0 || state.position.x > c.rmax+5.0 ||
         state.position.y < -y_radius-5.0 || state.position.y > y_radius+5.0 ||
         state.position.z < c.altitude-z_radius-5.0 || state.position.z > c.altitude+z_radius+5.0;
}

bool runSimulation(SolverKind solver_kind, const TrajectoryConfig& trajectory, double laps,
                   int repetition, std::ostream* csv, Aggregate* aggregate,
                   FirstSolve* first_solve) {
  const Parameters parameters;
  State state = referenceAt(trajectory, 0.0, parameters).state;
  // Table III contains references whose flatness-derived body rates exceed
  // the MPC's hard rate limits. Start from a feasible measured state so those
  // cases exercise bounded best-effort tracking instead of an infeasible
  // initial-condition QP.
  state.body_rate.x = std::clamp(state.body_rate.x, -10.0, 10.0);
  state.body_rate.y = std::clamp(state.body_rate.y, -10.0, 10.0);
  state.body_rate.z = std::clamp(state.body_rate.z, -4.0, 4.0);
  AcadosController controller(solver_kind);
  controller.initialize(makeHorizon(trajectory, 0.0));
  const double requested_steps = std::ceil(laps*trajectory.period/kControlDt - 1e-12);
  if (!std::isfinite(requested_steps) || requested_steps > std::numeric_limits<int>::max())
    throw std::invalid_argument("lap count exceeds supported simulation length");
  const int steps = std::max(1, static_cast<int>(requested_steps));
  bool valid_run = true;
  for (int step = 0; step < steps; ++step) {
    const double time = step*kControlDt;
    const auto horizon = makeHorizon(trajectory, time);
    controller.updateInitialState(state);
    controller.updateReferences(horizon);
    const SolveStats stats = controller.solve();
    const Control u = controller.control();
    const FlatReference& ref = horizon.front();
    const double position_error = norm(state.position-ref.state.position);
    const double attitude_error = attitudeErrorAngle(state.attitude, ref.state.attitude);
    const double velocity_error = norm(state.velocity-ref.state.velocity);

    bool applied_control_bounded = true;
    for (double value : u)
      applied_control_bounded = applied_control_bounded &&
        value >= parameters.thrust_min-1e-7 && value <= parameters.thrust_max+1e-7;
    const bool success = stats.status == 0 && stats.qp_status == 0 && allFinite(stats) &&
                         finite(u) && applied_control_bounded;
    if (csv) {
      writeStep(*csv, solver_kind, trajectory, repetition, step, time, stats, state, ref, u,
                position_error, attitude_error, velocity_error);
      csv->flush();
    }
    if (aggregate) {
      aggregate->samples++;
      aggregate->time_tot.push_back(stats.time_tot);
      aggregate->wall.push_back(stats.wall_time);
      aggregate->time_lin.push_back(stats.time_lin);
      aggregate->time_sim.push_back(stats.time_sim);
      aggregate->time_qp.push_back(stats.time_qp);
      aggregate->time_qp_solver.push_back(stats.time_qp_solver_call);
      aggregate->time_qp_xcond.push_back(stats.time_qp_xcond);
      aggregate->sampling_index.push_back(step);
      aggregate->position_error_sq += position_error*position_error;
      aggregate->attitude_error_sq += attitude_error*attitude_error;
      aggregate->velocity_error_sq += velocity_error*velocity_error;
      aggregate->max_position_error = std::max(aggregate->max_position_error, position_error);
      aggregate->max_attitude_error = std::max(aggregate->max_attitude_error, attitude_error);
      aggregate->max_bound_violation = std::max(aggregate->max_bound_violation, stats.max_bound_violation);
      aggregate->qp_iterations += static_cast<std::uint64_t>(std::max(0, stats.qp_iter));
      aggregate->max_qp_residual = std::max(
        aggregate->max_qp_residual, maxQpResidual(stats, solver_kind));
      for (double value : u) {
        if (value <= parameters.thrust_min+1e-5) aggregate->lower_active++;
        if (value >= parameters.thrust_max-1e-5) aggregate->upper_active++;
      }
      if (std::abs(state.body_rate.x) >= 10.0-1e-5 || std::abs(state.body_rate.y) >= 10.0-1e-5 ||
          std::abs(state.body_rate.z) >= 4.0-1e-5) aggregate->rate_active++;
      if (!success) {
        aggregate->failures++;
        aggregate->valid = false;
      }
    }
    if (first_solve && step == 0) {
      first_solve->control = u;
      first_solve->objective = stats.objective;
      first_solve->qp_res = stats.qp_residuals;
      first_solve->valid = success;
    }
    if (!success) {
      valid_run = false;
      break;
    }

    state = integratePlantRk4(state, u, kControlDt, parameters, kPlantDt);
    if (crashCondition(state, trajectory)) {
      std::cerr << "\nCrash condition for " << toString(solver_kind) << " / "
                << trajectory.name << " after step " << step
                << ": finite=" << finite(state) << ", position=("
                << state.position.x << ',' << state.position.y << ',' << state.position.z << ")\n";
      valid_run = false;
      if (aggregate) {
        aggregate->crashes++;
        aggregate->valid = false;
      }
      break;
    }
    controller.shiftWarmStart(referenceAt(
      trajectory, time+kControlDt+static_cast<double>(kN)*kHorizonDt, parameters));
  }
  if (aggregate && valid_run) aggregate->completed_runs++;
  return valid_run;
}

std::string aggregateTrajectory(const Aggregate& aggregate) {
  return aggregate.trajectory_label;
}

void writeSummaryCsv(const std::vector<Aggregate>& aggregates, const fs::path& file) {
  std::ofstream out(file);
  out << "solver,trajectory,valid,completed_runs,samples,averaged_sampling_points,failures,crashes,average_time_tot_us,max_per_sampling_average_time_tot_us,"
         "average_wall_us,average_time_lin_us,average_time_sim_us,average_time_qp_us,average_time_qp_solver_call_us,average_time_qp_xcond_us,"
         "position_rmse_m,max_position_error_m,attitude_rmse_rad,max_attitude_error_rad,velocity_rmse_mps,mean_qp_iterations,max_qp_residual,max_bound_violation,lower_active,upper_active,rate_active\n";
  out << std::setprecision(17);
  for (const auto& a : aggregates) {
    const double n = static_cast<double>(std::max<std::uint64_t>(1, a.samples));
    const auto averaged_time_tot = perSamplingAverages(a.time_tot, a.sampling_index);
    out << toString(a.solver) << ',' << aggregateTrajectory(a) << ',' << (a.valid ? "true" : "false") << ','
        << a.completed_runs << ',' << a.samples << ',' << averaged_time_tot.size() << ',' << a.failures << ',' << a.crashes << ','
        << 1e6*average(averaged_time_tot) << ',' << 1e6*maximum(averaged_time_tot) << ','
        << 1e6*average(perSamplingAverages(a.wall, a.sampling_index)) << ','
        << 1e6*average(perSamplingAverages(a.time_lin, a.sampling_index)) << ','
        << 1e6*average(perSamplingAverages(a.time_sim, a.sampling_index)) << ','
        << 1e6*average(perSamplingAverages(a.time_qp, a.sampling_index)) << ','
        << 1e6*average(perSamplingAverages(a.time_qp_solver, a.sampling_index)) << ','
        << 1e6*average(perSamplingAverages(a.time_qp_xcond, a.sampling_index)) << ','
        << std::sqrt(a.position_error_sq/n) << ',' << a.max_position_error << ','
        << std::sqrt(a.attitude_error_sq/n) << ',' << a.max_attitude_error << ','
        << std::sqrt(a.velocity_error_sq/n) << ',' << static_cast<double>(a.qp_iterations)/n << ','
        << a.max_qp_residual << ',' << a.max_bound_violation << ','
        << a.lower_active << ',' << a.upper_active << ',' << a.rate_active << '\n';
  }
}

void writeSummaryJson(const std::vector<Aggregate>& aggregates, const fs::path& file) {
  std::ofstream out(file);
  out << std::setprecision(17) << "{\n  \"results\": [\n";
  for (std::size_t i = 0; i < aggregates.size(); ++i) {
    const auto& a = aggregates[i];
    const double n = static_cast<double>(std::max<std::uint64_t>(1, a.samples));
    const auto averaged_time_tot = perSamplingAverages(a.time_tot, a.sampling_index);
    out << "    {\"solver\": \"" << toString(a.solver) << "\", \"trajectory\": \"" << aggregateTrajectory(a)
        << "\", \"valid\": " << (a.valid ? "true" : "false")
        << ", \"completed_runs\": " << a.completed_runs << ", \"samples\": " << a.samples
        << ", \"failures\": " << a.failures << ", \"crashes\": " << a.crashes
        << ", \"averaged_sampling_points\": " << averaged_time_tot.size()
        << ", \"time_tot_us\": {\"average\": " << 1e6*average(averaged_time_tot)
        << ", \"max_per_sampling_average\": " << 1e6*maximum(averaged_time_tot) << "}"
        << ", \"position_rmse_m\": " << std::sqrt(a.position_error_sq/n)
        << ", \"max_position_error_m\": " << a.max_position_error
        << ", \"max_qp_residual\": " << a.max_qp_residual
        << ", \"max_bound_violation\": " << a.max_bound_violation << "}";
    out << (i+1 == aggregates.size() ? "\n" : ",\n");
  }
  out << "  ]\n}\n";
}

bool checkConsistency(const std::map<std::pair<SolverKind,std::string>, FirstSolve>& first,
                      const Options& options, const fs::path& file) {
  bool valid = true;
  std::ofstream out(file);
  out << std::setprecision(17) << "{\n  \"comparisons\": [\n";
  bool first_item = true;
  for (const TrajectoryConfig& trajectory : options.trajectories) {
    const auto base_it = first.find({options.solvers.front(), trajectory.name});
    if (base_it == first.end() || !base_it->second.valid) { valid = false; continue; }
    for (std::size_t i = 1; i < options.solvers.size(); ++i) {
      const auto other_it = first.find({options.solvers[i], trajectory.name});
      if (other_it == first.end() || !other_it->second.valid) { valid = false; continue; }
      double control_diff = 0.0;
      for (int j = 0; j < kNu; ++j) control_diff = std::max(control_diff,
        std::abs(base_it->second.control[static_cast<std::size_t>(j)]-other_it->second.control[static_cast<std::size_t>(j)]));
      const double objective_scale = std::max({1.0, std::abs(base_it->second.objective), std::abs(other_it->second.objective)});
      const double objective_relative = std::abs(base_it->second.objective-other_it->second.objective)/objective_scale;
      double residual_max = maxQpResidual(base_it->second.qp_res, options.solvers.front());
      residual_max = std::max(
        residual_max, maxQpResidual(other_it->second.qp_res, options.solvers[i]));
      const bool pair_valid = control_diff <= 5e-3 && objective_relative <= 2e-3;
      valid = valid && pair_valid;
      if (!first_item) out << ",\n";
      first_item = false;
      out << "    {\"trajectory\": \"" << toString(trajectory) << "\", \"baseline\": \""
          << toString(options.solvers.front()) << "\", \"solver\": \"" << toString(options.solvers[i])
          << "\", \"max_control_abs_diff\": " << control_diff
          << ", \"objective_relative_diff\": " << objective_relative
          << ", \"max_qp_residual\": " << residual_max
          << ", \"valid\": " << (pair_valid ? "true" : "false") << "}";
    }
  }
  out << "\n  ],\n  \"valid\": " << (valid ? "true" : "false") << "\n}\n";
  return valid;
}

void printTable(const std::vector<Aggregate>& aggregates) {
  std::cout << "\nSolver timing and tracking comparison (acados time_tot)\n"
            << std::left << std::setw(11) << "solver" << std::setw(14) << "trajectory"
            << std::right << std::setw(15) << "average [us]"
            << std::setw(15) << "worst [us]" << std::setw(12) << "pos RMSE" << std::setw(9) << "valid" << '\n';
  for (const auto& a : aggregates) {
    const double n = static_cast<double>(std::max<std::uint64_t>(1, a.samples));
    const auto averaged_time_tot = perSamplingAverages(a.time_tot, a.sampling_index);
    std::cout << std::left << std::setw(11) << toString(a.solver) << std::setw(14) << aggregateTrajectory(a)
              << std::right << std::fixed << std::setprecision(2)
              << std::setw(15) << 1e6*average(averaged_time_tot)
              << std::setw(15) << 1e6*maximum(averaged_time_tot) << std::setw(12) << std::setprecision(4)
              << std::sqrt(a.position_error_sq/n) << std::setw(9) << (a.valid ? "yes" : "NO") << '\n';
  }
}

void printGridProgress(std::size_t completed, std::size_t total) {
  constexpr std::size_t width = 30;
  const std::size_t filled = width*completed/total;
  std::cout << '[' << std::string(filled, '=') << std::string(width-filled, ' ')
            << "] " << completed << '/' << total << " trajectories completed ("
            << 100*completed/total << "%)\n" << std::flush;
}

int generatePlots(const fs::path& output) {
  const fs::path script = fs::path(BENCH_SOURCE_DIR) / "tools" / "plot_results.py";
  const pid_t pid = fork();
  if (pid < 0) return -1;
  if (pid == 0) {
    execl(BENCH_PYTHON_EXECUTABLE, BENCH_PYTHON_EXECUTABLE, script.c_str(),
          "--input", output.c_str(), static_cast<char*>(nullptr));
    _exit(127);
  }
  int status = 0;
  if (waitpid(pid, &status, 0) < 0) return -1;
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

}  // namespace
}  // namespace quad

int main(int argc, char** argv) {
  using namespace quad;
  try {
    const Options options = parseOptions(argc, argv);
    applyAffinity(options.cpu_affinity);
    for (auto solver : options.solvers)
      if (!solverAvailable(solver))
        throw std::runtime_error(toString(solver) + " is not enabled in the installed acados library");
    std::ofstream csv;
    if (!kGridBenchmark) {
      fs::create_directories(options.output);
      writeMetadata(options, options.output / "metadata.json");
      csv.open(options.output / "steps.csv");
      if (!csv) throw std::runtime_error("cannot create steps.csv");
      writeCsvHeader(csv);
    }
    std::vector<Aggregate> reported;
    std::map<SolverKind, double> average_wins, worst_wins;
    std::size_t comparable = 0;
    std::size_t completed_trajectories = 0;
    bool all_valid = true;
    if (kGridBenchmark) printGridProgress(0, options.trajectories.size());
    for (const auto& trajectory : options.trajectories) {
      std::vector<Aggregate> results;
      std::map<std::pair<SolverKind,std::string>, FirstSolve> first_solves;
      for (SolverKind solver : options.solvers) {
        if (!kGridBenchmark)
          std::cerr << "Running " << toString(solver) << " / " << trajectory.name << std::flush;
        Aggregate aggregate;
        aggregate.solver = solver;
        aggregate.trajectory_label = trajectory.name;
        try {
          if (options.warmup && !runSimulation(solver, trajectory, options.laps, -1, nullptr, nullptr, nullptr))
            std::cerr << "Warm-up failed for " << toString(solver) << " / " << trajectory.name << '\n';
          for (int repetition = 0; repetition < options.repetitions; ++repetition) {
            FirstSolve* first = repetition == 0 ? &first_solves[{solver,trajectory.name}] : nullptr;
            runSimulation(solver, trajectory, options.laps, repetition,
                          kGridBenchmark ? nullptr : &csv, &aggregate, first);
            if (!kGridBenchmark) std::cerr << '.' << std::flush;
          }
        } catch (const std::exception& e) {
          aggregate.valid = false;
          ++aggregate.failures;
          std::cerr << "Run failed for " << toString(solver) << " / " << trajectory.name
                    << ": " << e.what() << '\n';
        }
        aggregate.valid = aggregate.valid && aggregate.completed_runs == options.repetitions;
        all_valid = all_valid && aggregate.valid;
        results.push_back(std::move(aggregate));
        if (!kGridBenchmark) std::cerr << '\n';
      }
      if (kGridBenchmark) {
        ++completed_trajectories;
        if (completed_trajectories % 5 == 0 || completed_trajectories == options.trajectories.size())
          printGridProgress(completed_trajectories, options.trajectories.size());
        const bool complete = std::all_of(results.begin(), results.end(),
          [](const Aggregate& a) { return a.valid; });
        if (!complete) {
          std::cerr << "Excluding " << trajectory.name << ": at least one solver did not complete every run.\n";
          continue;
        }
        ++comparable;
        std::vector<double> means, maxima;
        for (const auto& a : results) {
          const auto trace = perSamplingAverages(a.time_tot, a.sampling_index);
          means.push_back(average(trace));
          maxima.push_back(maximum(trace));
        }
        const auto award = [&](const std::vector<double>& times, std::map<SolverKind, double>& wins) {
          const auto shares = fastestShares(times);
          for (std::size_t i = 0; i < times.size(); ++i) wins[results[i].solver] += shares[i];
        };
        award(means, average_wins);
        award(maxima, worst_wins);
      } else {
        if (options.solvers.size() > 1)
          all_valid = checkConsistency(first_solves, options, options.output / "consistency.json") && all_valid;
        reported = std::move(results);
      }
    }
    if (kGridBenchmark) {
      std::cout << "Comparable trajectories: " << comparable << '/' << options.trajectories.size()
                << " (failed comparisons excluded; exact ties share a win)\n"
                << std::left << std::setw(12) << "solver" << std::right
                << std::setw(20) << "fastest average [%]" << std::setw(23) << "fastest worst case [%]" << '\n';
      for (auto solver : options.solvers) {
        const double denominator = comparable ? static_cast<double>(comparable) :
          std::numeric_limits<double>::quiet_NaN();
        std::cout << std::left << std::setw(12) << toString(solver) << std::right
                  << std::fixed << std::setprecision(2)
                  << std::setw(20) << 100.0*average_wins[solver]/denominator
                  << std::setw(23) << 100.0*worst_wins[solver]/denominator << '\n';
      }
    } else {
      csv.close();
      writeSummaryCsv(reported, options.output / "summary.csv");
      writeSummaryJson(reported, options.output / "summary.json");
      printTable(reported);
      if (options.plots && generatePlots(options.output) != 0)
        throw std::runtime_error("plot generation failed");
      std::cout << "\nArtifacts: " << fs::absolute(options.output) << '\n';
    }
    return all_valid ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << (kGridBenchmark ? "quadrotor_grid_benchmark: " : "quadrotor_benchmark: ") << e.what() << '\n';
    return 1;
  }
}
