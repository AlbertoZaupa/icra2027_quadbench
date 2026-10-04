#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "quadrotor/trajectory.hpp"

namespace quad {

enum class SolverKind { Daocp, Hpipm, Daqp, Qpoases };

std::string toString(SolverKind kind);
SolverKind parseSolver(const std::string& name);
bool solverAvailable(SolverKind kind);

struct SolveStats {
  int status{-1};
  int qp_status{-1};
  int sqp_iter{};
  int qp_iter{};
  double time_tot{};
  double time_lin{};
  double time_sim{};
  double time_qp{};
  double time_qp_solver_call{};
  double time_qp_xcond{};
  double wall_time{};
  double objective{};
  double max_bound_violation{};
  std::array<double, 4> nlp_residuals{};
  std::array<double, 4> qp_residuals{};
};

class AcadosController {
 public:
  explicit AcadosController(SolverKind solver);
  ~AcadosController();
  AcadosController(const AcadosController&) = delete;
  AcadosController& operator=(const AcadosController&) = delete;

  void initialize(const std::vector<FlatReference>& horizon);
  void updateInitialState(const State& state);
  void updateReferences(const std::vector<FlatReference>& horizon);
  SolveStats solve();
  Control control() const;
  void shiftWarmStart(const FlatReference& next_terminal_reference);
  SolverKind kind() const { return solver_kind_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  SolverKind solver_kind_;
};

}  // namespace quad
