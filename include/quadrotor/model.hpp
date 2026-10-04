#pragma once

#include <array>
#include "quadrotor/math.hpp"

namespace quad {

constexpr int kNx = 13;
constexpr int kNu = 4;
constexpr int kN = 20;
constexpr double kHorizonDt = 0.05;
constexpr double kControlDt = 0.01;
constexpr double kPlantDt = 0.002;

struct Parameters {
  double mass{0.75};
  double arm_length{0.14};
  double beta{56.0 * kPi / 180.0};
  Vec3 inertia{0.0025, 0.0021, 0.0043};
  double thrust_min{0.0};
  double thrust_max{8.5};
  double torque_coefficient{2.37e-8};
  double thrust_coefficient{1.51e-6};
  Vec3 linear_drag{0.26, 0.28, 0.42};
  double horizontal_drag{0.018};
  double gravity{9.81};
};

struct State {
  Vec3 position{};
  Quaternion attitude{};
  Vec3 velocity{};
  Vec3 body_rate{};
};

using Control = std::array<double, kNu>;

struct Derivative {
  Vec3 position{};
  Quaternion attitude{0.0, 0.0, 0.0, 0.0};
  Vec3 velocity{};
  Vec3 body_rate{};
};

std::array<double, kNx> toArray(const State& state);
State stateFromArray(const double* x);
std::array<double, kNx> toArray(const Derivative& derivative);
Derivative dynamics(const State& state, const Control& control,
                    const Parameters& parameters = {});
Vec3 bodyDrag(const State& state, const Parameters& parameters = {});
Vec3 rotorTorque(const Control& control, const Parameters& parameters = {});
State integratePlantRk4(const State& state, const Control& control, double duration,
                        const Parameters& parameters = {}, double max_step = kPlantDt);
double attitudeErrorAngle(const Quaternion& a, const Quaternion& b);
bool finite(const State& state);
bool finite(const Control& control);

}  // namespace quad
