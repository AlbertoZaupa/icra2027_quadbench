#pragma once

#include <string>
#include <vector>
#include "quadrotor/model.hpp"

namespace quad {

enum class TrajectoryKind { Challenging, DaocpWorst };
enum class TrajectoryPlane { Horizontal, Vertical };

struct TrajectoryConfig {
  TrajectoryKind kind{TrajectoryKind::Challenging};
  TrajectoryPlane plane{TrajectoryPlane::Horizontal};
  std::string name{"challenging"};
  double vmax{15.0};
  double amax{30.0};
  double ellipticity{2.0};
  double rmax{7.5};
  double rmin{3.75};
  double rate{2.0};
  double period{kPi};
  double altitude{5.0};
};

struct FlatReference {
  State state{};
  Control control{};
  Vec3 acceleration{};
  Vec3 jerk{};
  double heading{};
  double heading_rate{};
  double collective_thrust{};
};

TrajectoryConfig trajectoryConfig(TrajectoryKind kind);
TrajectoryConfig makeTrajectoryConfig(TrajectoryPlane plane, double vmax,
                                      double amax, double ellipticity);
std::vector<TrajectoryConfig> paperTrajectoryGrid();
TrajectoryConfig parseTrajectory(const std::string& name);
std::string toString(TrajectoryKind kind);
std::string toString(const TrajectoryConfig& config);
FlatReference referenceAt(TrajectoryKind kind, double time,
                          const Parameters& parameters = {});
FlatReference referenceAt(const TrajectoryConfig& config, double time,
                          const Parameters& parameters = {});

}  // namespace quad
