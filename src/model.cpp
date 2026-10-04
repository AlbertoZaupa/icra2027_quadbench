#include "quadrotor/model.hpp"

#include <algorithm>
#include <cmath>

namespace quad {

Mat3 rotationMatrix(const Quaternion& q_in) {
  // Do not normalize here: this is also the controller model used by the ODE.
  // Unit norm is preserved by the continuous quaternion dynamics and the plant
  // explicitly normalizes after every RK4 step.
  const Quaternion& q = q_in;
  const double ww = q.w*q.w;
  const double xx = q.x*q.x;
  const double yy = q.y*q.y;
  const double zz = q.z*q.z;
  return {
    {ww + xx - yy - zz, 2.0*(q.x*q.y + q.w*q.z), 2.0*(q.x*q.z - q.w*q.y)},
    {2.0*(q.x*q.y - q.w*q.z), ww - xx + yy - zz, 2.0*(q.y*q.z + q.w*q.x)},
    {2.0*(q.x*q.z + q.w*q.y), 2.0*(q.y*q.z - q.w*q.x), ww - xx - yy + zz}
  };
}

Quaternion quaternionFromRotation(const Mat3& r) {
  // Convert column-major R to scalar-first quaternion. The branch result is
  // sign-ambiguous; trajectory.cpp performs continuous sign lifting.
  const double r00 = r.x.x, r01 = r.y.x, r02 = r.z.x;
  const double r10 = r.x.y, r11 = r.y.y, r12 = r.z.y;
  const double r20 = r.x.z, r21 = r.y.z, r22 = r.z.z;
  Quaternion q;
  const double trace = r00 + r11 + r22;
  if (trace > 0.0) {
    const double s = 2.0 * std::sqrt(std::max(0.0, trace + 1.0));
    q = {0.25*s, (r21-r12)/s, (r02-r20)/s, (r10-r01)/s};
  } else if (r00 > r11 && r00 > r22) {
    const double s = 2.0 * std::sqrt(std::max(0.0, 1.0+r00-r11-r22));
    q = {(r21-r12)/s, 0.25*s, (r01+r10)/s, (r02+r20)/s};
  } else if (r11 > r22) {
    const double s = 2.0 * std::sqrt(std::max(0.0, 1.0+r11-r00-r22));
    q = {(r02-r20)/s, (r01+r10)/s, 0.25*s, (r12+r21)/s};
  } else {
    const double s = 2.0 * std::sqrt(std::max(0.0, 1.0+r22-r00-r11));
    q = {(r10-r01)/s, (r02+r20)/s, (r12+r21)/s, 0.25*s};
  }
  return normalized(q);
}

std::array<double, kNx> toArray(const State& s) {
  return {s.position.x, s.position.y, s.position.z,
          s.attitude.w, s.attitude.x, s.attitude.y, s.attitude.z,
          s.velocity.x, s.velocity.y, s.velocity.z,
          s.body_rate.x, s.body_rate.y, s.body_rate.z};
}

State stateFromArray(const double* x) {
  return {{x[0], x[1], x[2]}, {x[3], x[4], x[5], x[6]},
          {x[7], x[8], x[9]}, {x[10], x[11], x[12]}};
}

std::array<double, kNx> toArray(const Derivative& d) {
  return {d.position.x, d.position.y, d.position.z,
          d.attitude.w, d.attitude.x, d.attitude.y, d.attitude.z,
          d.velocity.x, d.velocity.y, d.velocity.z,
          d.body_rate.x, d.body_rate.y, d.body_rate.z};
}

Vec3 bodyDrag(const State& state, const Parameters& p) {
  const Vec3 vb = transposeMultiply(rotationMatrix(state.attitude), state.velocity);
  return {-p.linear_drag.x * vb.x,
          -p.linear_drag.y * vb.y,
          -p.linear_drag.z * vb.z + p.horizontal_drag * (vb.x*vb.x + vb.y*vb.y)};
}

Vec3 rotorTorque(const Control& u, const Parameters& p) {
  const double lx = p.arm_length * std::sin(p.beta);
  const double ly = p.arm_length * std::cos(p.beta);
  const double kappa = p.torque_coefficient / p.thrust_coefficient;
  return {lx*(u[0]-u[1]-u[2]+u[3]),
          ly*(-u[0]-u[1]+u[2]+u[3]),
          kappa*(u[0]-u[1]+u[2]-u[3])};
}

Derivative dynamics(const State& state, const Control& control, const Parameters& p) {
  const Mat3 r = rotationMatrix(state.attitude);
  const double thrust = control[0] + control[1] + control[2] + control[3];
  const Vec3 force_world = r * (bodyDrag(state, p) + Vec3{0.0, 0.0, thrust});
  const Vec3 acceleration = force_world / p.mass + Vec3{0.0, 0.0, -p.gravity};

  const Quaternion omega_q{0.0, state.body_rate.x, state.body_rate.y, state.body_rate.z};
  const Quaternion qdot = multiply(state.attitude, omega_q) * 0.5;

  const Vec3 iw{p.inertia.x*state.body_rate.x,
                p.inertia.y*state.body_rate.y,
                p.inertia.z*state.body_rate.z};
  const Vec3 rhs = rotorTorque(control, p) - cross(state.body_rate, iw);
  const Vec3 wdot{rhs.x/p.inertia.x, rhs.y/p.inertia.y, rhs.z/p.inertia.z};
  return {state.velocity, qdot, acceleration, wdot};
}

namespace {

State addScaled(const State& s, const Derivative& d, double h) {
  return {s.position + d.position*h,
          s.attitude + d.attitude*h,
          s.velocity + d.velocity*h,
          s.body_rate + d.body_rate*h};
}

State rk4Step(const State& s, const Control& u, double h, const Parameters& p) {
  const Derivative k1 = dynamics(s, u, p);
  const Derivative k2 = dynamics(addScaled(s, k1, 0.5*h), u, p);
  const Derivative k3 = dynamics(addScaled(s, k2, 0.5*h), u, p);
  const Derivative k4 = dynamics(addScaled(s, k3, h), u, p);
  State out = s;
  const auto add = [h](double v, double a, double b, double c, double d) {
    return v + (h/6.0)*(a + 2.0*b + 2.0*c + d);
  };
  out.position = {
    add(s.position.x,k1.position.x,k2.position.x,k3.position.x,k4.position.x),
    add(s.position.y,k1.position.y,k2.position.y,k3.position.y,k4.position.y),
    add(s.position.z,k1.position.z,k2.position.z,k3.position.z,k4.position.z)};
  out.attitude = {
    add(s.attitude.w,k1.attitude.w,k2.attitude.w,k3.attitude.w,k4.attitude.w),
    add(s.attitude.x,k1.attitude.x,k2.attitude.x,k3.attitude.x,k4.attitude.x),
    add(s.attitude.y,k1.attitude.y,k2.attitude.y,k3.attitude.y,k4.attitude.y),
    add(s.attitude.z,k1.attitude.z,k2.attitude.z,k3.attitude.z,k4.attitude.z)};
  out.velocity = {
    add(s.velocity.x,k1.velocity.x,k2.velocity.x,k3.velocity.x,k4.velocity.x),
    add(s.velocity.y,k1.velocity.y,k2.velocity.y,k3.velocity.y,k4.velocity.y),
    add(s.velocity.z,k1.velocity.z,k2.velocity.z,k3.velocity.z,k4.velocity.z)};
  out.body_rate = {
    add(s.body_rate.x,k1.body_rate.x,k2.body_rate.x,k3.body_rate.x,k4.body_rate.x),
    add(s.body_rate.y,k1.body_rate.y,k2.body_rate.y,k3.body_rate.y,k4.body_rate.y),
    add(s.body_rate.z,k1.body_rate.z,k2.body_rate.z,k3.body_rate.z,k4.body_rate.z)};
  out.attitude = normalized(out.attitude);
  return out;
}

}  // namespace

State integratePlantRk4(const State& state, const Control& control, double duration,
                        const Parameters& p, double max_step) {
  if (duration < 0.0 || max_step <= 0.0) throw std::invalid_argument("invalid integration interval");
  State result = state;
  double remaining = duration;
  while (remaining > 1e-15) {
    const double h = std::min(max_step, remaining);
    result = rk4Step(result, control, h, p);
    remaining -= h;
  }
  return result;
}

double attitudeErrorAngle(const Quaternion& a_in, const Quaternion& b_in) {
  const Quaternion a = normalized(a_in);
  const Quaternion b = normalized(b_in);
  return 2.0 * std::acos(clamp(std::abs(dot(a, b)), 0.0, 1.0));
}

bool finite(const State& s) {
  const auto a = toArray(s);
  return std::all_of(a.begin(), a.end(), [](double v) { return std::isfinite(v); });
}

bool finite(const Control& u) {
  return std::all_of(u.begin(), u.end(), [](double v) { return std::isfinite(v); });
}

}  // namespace quad

