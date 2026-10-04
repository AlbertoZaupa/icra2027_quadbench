#include "quadrotor/trajectory.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace quad {

TrajectoryConfig trajectoryConfig(TrajectoryKind kind) {
  TrajectoryConfig c;
  c.kind = kind;
  if (kind == TrajectoryKind::Challenging) {
    c.name = "challenging";
    c.vmax = 15.0;
    c.amax = 30.0;
    c.ellipticity = 2.0;
  } else if (kind == TrajectoryKind::DaocpWorst) {
    c.name = "daocp-worst";
    c.vmax = 15.0;
    c.amax = 60.0;
    c.ellipticity = 1.0;
  } else {
    throw std::invalid_argument("invalid trajectory kind");
  }
  c.rmax = c.vmax*c.vmax/c.amax;
  c.rmin = c.rmax/c.ellipticity;
  c.rate = c.amax/c.vmax;
  c.period = 2.0*kPi/c.rate;
  return c;
}

TrajectoryConfig makeTrajectoryConfig(TrajectoryPlane plane, double vmax,
                                      double amax, double ellipticity) {
  if (!(vmax > 0.0) || !(amax > 0.0) || !(ellipticity > 0.0))
    throw std::invalid_argument("trajectory parameters must be positive");
  TrajectoryConfig c;
  c.plane = plane;
  c.vmax = vmax;
  c.amax = amax;
  c.ellipticity = ellipticity;
  c.rmax = c.vmax*c.vmax/c.amax;
  c.rmin = c.rmax/c.ellipticity;
  c.rate = c.amax/c.vmax;
  c.period = 2.0*kPi/c.rate;
  std::ostringstream label;
  label << (plane == TrajectoryPlane::Horizontal ? "horizontal" : "vertical")
        << "-a" << std::fixed << std::setprecision(0) << amax
        << "-v" << vmax << "-n" << ellipticity;
  c.name = label.str();
  return c;
}

std::vector<TrajectoryConfig> paperTrajectoryGrid() {
  std::vector<TrajectoryConfig> result;
  result.reserve(144);
  for (TrajectoryPlane plane : {TrajectoryPlane::Horizontal, TrajectoryPlane::Vertical})
    for (double amax : {10.0, 20.0, 30.0, 40.0, 50.0, 60.0})
      for (double ellipticity : {1.0, 2.0, 5.0})
        for (double vmax : {5.0, 10.0, 15.0, 20.0})
          result.push_back(makeTrajectoryConfig(plane, vmax, amax, ellipticity));
  return result;
}

TrajectoryConfig parseTrajectory(const std::string& name) {
  if (name == "challenging" || name == "hard") return trajectoryConfig(TrajectoryKind::Challenging);
  if (name == "daocp-worst" || name == "daocp_worst") return trajectoryConfig(TrajectoryKind::DaocpWorst);
  for (const auto& config : paperTrajectoryGrid()) if (config.name == name) return config;
  throw std::invalid_argument("unknown trajectory: " + name);
}

std::string toString(TrajectoryKind kind) {
  return trajectoryConfig(kind).name;
}

std::string toString(const TrajectoryConfig& config) { return config.name; }

namespace {

double unwrappedHeading(double theta, double rmax, double rmin) {
  const double raw = std::atan2(-rmin*std::cos(theta), -rmax*std::sin(theta));
  // Select the equivalent angle nearest the circle's continuous inward
  // heading. This remains well-defined at atan2's branch cut despite
  // roundoff in sin/cos at odd multiples of pi/2.
  const double nominal = -theta - 0.5*kPi;
  return raw + 2.0*kPi*std::round((nominal-raw)/(2.0*kPi));
}

}  // namespace

FlatReference referenceAt(TrajectoryKind kind, double time, const Parameters& p) {
  return referenceAt(trajectoryConfig(kind), time, p);
}

FlatReference referenceAt(const TrajectoryConfig& c, double time, const Parameters& p) {
  const double theta = c.rate*time;
  const double st = std::sin(theta);
  const double ct = std::cos(theta);

  const bool horizontal = c.plane == TrajectoryPlane::Horizontal;
  const Vec3 position = horizontal
    ? Vec3{c.rmax*st, c.rmin*ct, c.altitude}
    : Vec3{c.rmax*st, 0.0, c.altitude+c.rmin*ct};
  const Vec3 velocity = horizontal
    ? Vec3{c.rmax*c.rate*ct, -c.rmin*c.rate*st, 0.0}
    : Vec3{c.rmax*c.rate*ct, 0.0, -c.rmin*c.rate*st};
  const Vec3 acceleration = horizontal
    ? Vec3{-c.rmax*c.rate*c.rate*st, -c.rmin*c.rate*c.rate*ct, 0.0}
    : Vec3{-c.rmax*c.rate*c.rate*st, 0.0, -c.rmin*c.rate*c.rate*ct};
  const Vec3 jerk = horizontal
    ? Vec3{-c.rmax*c.rate*c.rate*c.rate*ct, c.rmin*c.rate*c.rate*c.rate*st, 0.0}
    : Vec3{-c.rmax*c.rate*c.rate*c.rate*ct, 0.0, c.rmin*c.rate*c.rate*c.rate*st};

  // The heading points toward the center. Using axes instead of wrapped yaw
  // keeps the attitude construction analytic through the atan2 branch cut.
  const Vec3 center_direction{-position.x, -position.y, 0.0};
  const Vec3 center_velocity{-velocity.x, -velocity.y, 0.0};
  const Vec3 x_c = horizontal ? normalized(center_direction) : Vec3{1.0, 0.0, 0.0};
  const Vec3 x_c_dot = horizontal
    ? normalizedDerivative(center_direction, center_velocity) : Vec3{};
  const Vec3 y_c{-x_c.y, x_c.x, 0.0};
  const Vec3 y_c_dot{-x_c_dot.y, x_c_dot.x, 0.0};

  // Differential flatness with axis-dependent rotor drag (Faessler et al.).
  // The paper's additional horizontal-drag term acts only along body z and
  // therefore changes collective thrust, not the attitude axes.
  const Vec3 gravity_comp{0.0, 0.0, p.gravity};
  const Vec3 alpha = acceleration + gravity_comp + velocity*(p.linear_drag.x/p.mass);
  const Vec3 alpha_dot = jerk + acceleration*(p.linear_drag.x/p.mass);
  const Vec3 beta = acceleration + gravity_comp + velocity*(p.linear_drag.y/p.mass);
  const Vec3 beta_dot = jerk + acceleration*(p.linear_drag.y/p.mass);

  const Vec3 x_tmp = cross(y_c, alpha);
  const Vec3 x_tmp_dot = cross(y_c_dot, alpha) + cross(y_c, alpha_dot);
  const Vec3 x_b = normalized(x_tmp);
  const Vec3 x_b_dot = normalizedDerivative(x_tmp, x_tmp_dot);

  const Vec3 y_tmp = cross(beta, x_b);
  const Vec3 y_tmp_dot = cross(beta_dot, x_b) + cross(beta, x_b_dot);
  const Vec3 y_b = normalized(y_tmp);
  const Vec3 y_b_dot = normalizedDerivative(y_tmp, y_tmp_dot);
  const Vec3 z_b = cross(x_b, y_b);
  const Vec3 z_b_dot = cross(x_b_dot, y_b) + cross(x_b, y_b_dot);
  const Mat3 rotation{x_b, y_b, z_b};

  Quaternion q = quaternionFromRotation(rotation);
  const double heading = horizontal ? unwrappedHeading(theta, c.rmax, c.rmin) : 0.0;
  Quaternion sign_anchor{std::cos(0.5*heading), 0.0, 0.0, std::sin(0.5*heading)};
  if (!horizontal) {
    const double raw_pitch = std::atan2(alpha.x, alpha.z);
    const double vertical_force_amplitude =
      std::hypot(c.amax, c.vmax*(p.linear_drag.x/p.mass))/c.ellipticity;
    const double pitch = vertical_force_amplitude > p.gravity
      ? raw_pitch + 2.0*kPi*std::round((theta+kPi-raw_pitch)/(2.0*kPi))
      : raw_pitch;
    sign_anchor = {std::cos(0.5*pitch), 0.0, std::sin(0.5*pitch), 0.0};
  }
  if (dot(q, sign_anchor) < 0.0) q = -q;

  // Rdot = R [omega]_x. The symmetric expressions suppress roundoff in the
  // orthonormality identities while retaining the analytic derivatives.
  const Vec3 omega{
    0.5*(dot(z_b, y_b_dot) - dot(y_b, z_b_dot)),
    0.5*(dot(x_b, z_b_dot) - dot(z_b, x_b_dot)),
    0.5*(dot(y_b, x_b_dot) - dot(x_b, y_b_dot))};

  const Vec3 vb = transposeMultiply(rotation, velocity);
  const double collective = p.mass*dot(z_b, acceleration + gravity_comp)
                          + p.linear_drag.z*vb.z
                          - p.horizontal_drag*(vb.x*vb.x + vb.y*vb.y);
  const double each = collective/4.0;

  const double heading_rate = horizontal ? -c.rmax*c.rmin*c.rate /
    (c.rmax*c.rmax*st*st + c.rmin*c.rmin*ct*ct) : 0.0;

  return {{position, q, velocity, omega}, {each, each, each, each},
          acceleration, jerk, heading, heading_rate, collective};
}

}  // namespace quad
