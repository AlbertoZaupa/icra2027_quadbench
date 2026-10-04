#include "quadrotor/model.hpp"
#include "quadrotor/trajectory.hpp"
#include "quadrotor/timing.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include "quadrotor_generated.h"
}

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void near(double actual, double expected, double tolerance, const std::string& message) {
  if (!std::isfinite(actual) || std::abs(actual-expected) > tolerance) {
    throw std::runtime_error(message + ": got " + std::to_string(actual) +
                             ", expected " + std::to_string(expected));
  }
}

void testTiming() {
  using namespace quad;
  // Two repeats whose slowest samples occur at different simulation steps.
  const auto trace = perSamplingAverages({2.0, 10.0, 8.0, 4.0}, {0, 1, 0, 1});
  require(trace == std::vector<double>({5.0, 7.0}), "timings grouped by sampling index");
  near(average(trace), 6.0, 1e-14, "mean of averaged trace");
  near(maximum(trace), 7.0, 1e-14, "worst case must follow per-step averaging");
  require(fastestShares({3.0, 1.0, 2.0}) == std::vector<double>({0.0, 1.0, 0.0}),
          "unique timing winner");
  require(fastestShares({1.0, 2.0, 1.0}) == std::vector<double>({0.5, 0.0, 0.5}),
          "tied solvers share a win");
  require(fastestShares({1.0}) == std::vector<double>({1.0}), "single solver wins");
  bool rejected = false;
  try { parseTrajectory("easy"); } catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "easy trajectory alias removed");
}

void testTrajectory(const quad::TrajectoryConfig& c) {
  using namespace quad;
  for (double t : {0.0, 0.137, 0.91, 2.73}) {
    const auto a = referenceAt(c, t);
    const auto b = referenceAt(c, t+c.period);
    near(norm(a.state.position-b.state.position), 0.0, 2e-12, "position periodicity");
    near(norm(a.state.velocity-b.state.velocity), 0.0, 1e-11, "velocity periodicity");
    near(norm(a.acceleration-b.acceleration), 0.0, 3e-11, "acceleration periodicity");
    near(norm(a.jerk-b.jerk), 0.0, 1e-10, "jerk periodicity");
    near(std::abs(dot(a.state.attitude,b.state.attitude)), 1.0, 2e-12, "attitude periodicity");
    near(norm(a.state.attitude), 1.0, 2e-13, "unit quaternion");
  }

  double vmax = 0.0;
  double amax = 0.0;
  double max_heading_rate = 0.0;
  Quaternion previous = referenceAt(c, -c.period).state.attitude;
  constexpr int samples = 20000;
  for (int i = 0; i <= samples*3; ++i) {
    const double t = -c.period + 3.0*c.period*static_cast<double>(i)/samples/3.0;
    const auto r = referenceAt(c, t);
    vmax = std::max(vmax, norm(r.state.velocity));
    amax = std::max(amax, norm(r.acceleration));
    max_heading_rate = std::max(max_heading_rate, std::abs(r.heading_rate));
    near(norm(r.state.attitude), 1.0, 3e-13, "quaternion norm over trajectory");
    if (i > 0 && !(dot(previous, r.state.attitude) > 0.999)) {
      throw std::runtime_error(c.name + " quaternion sign discontinuity at t=" + std::to_string(t) +
                               ", dot=" + std::to_string(dot(previous, r.state.attitude)));
    }
    previous = r.state.attitude;
  }
  near(vmax, c.vmax, 2e-7, "requested maximum speed");
  near(amax, c.amax, 2e-7, "requested maximum acceleration");
  if (c.name == "challenging" || c.name == "daocp-worst")
    near(max_heading_rate, 4.0, 2e-7, "yaw-rate boundary");
  if (c.plane == TrajectoryPlane::Vertical)
    near(max_heading_rate, 0.0, 1e-14, "vertical trajectory constant heading");

  const double h = 1e-6;
  for (double t : {0.07, 0.51, 1.21}) {
    const auto before = referenceAt(c, t-h);
    const auto center = referenceAt(c, t);
    const auto after = referenceAt(c, t+h);
    const Vec3 p_dot = (after.state.position-before.state.position)/(2.0*h);
    const Vec3 v_dot = (after.state.velocity-before.state.velocity)/(2.0*h);
    const Vec3 a_dot = (after.acceleration-before.acceleration)/(2.0*h);
    near(norm(p_dot-center.state.velocity), 0.0, 2e-7, "analytic velocity");
    near(norm(v_dot-center.acceleration), 0.0, 2e-6, "analytic acceleration");
    near(norm(a_dot-center.jerk), 0.0, 2e-5, "analytic jerk");

    Quaternion q_before = before.state.attitude;
    Quaternion q_after = after.state.attitude;
    if (dot(q_before, center.state.attitude) < 0.0) q_before = -q_before;
    if (dot(q_after, center.state.attitude) < 0.0) q_after = -q_after;
    const Quaternion qdot = (q_after + q_before*(-1.0))*(1.0/(2.0*h));
    const Quaternion omega_q = multiply(conjugate(center.state.attitude), qdot)*2.0;
    near(norm(Vec3{omega_q.x,omega_q.y,omega_q.z}-center.state.body_rate), 0.0, 3e-5,
         "analytic body rate");
  }
}

void testPaperTrajectoryGrid() {
  using namespace quad;
  const auto grid = paperTrajectoryGrid();
  require(grid.size() == 144, "Table III trajectory count");
  std::vector<std::string> names;
  for (const auto& trajectory : grid) names.push_back(trajectory.name);
  std::sort(names.begin(), names.end());
  require(std::adjacent_find(names.begin(), names.end()) == names.end(),
          "Table III trajectory names must be unique");
  testTrajectory(makeTrajectoryConfig(TrajectoryPlane::Horizontal, 20.0, 60.0, 5.0));
  testTrajectory(makeTrajectoryConfig(TrajectoryPlane::Vertical, 20.0, 60.0, 5.0));
}

void testModel() {
  using namespace quad;
  const Parameters p;
  near(p.horizontal_drag, 0.018, 1e-15, "Table II horizontal-drag coefficient");
  State hover;
  const double each = p.mass*p.gravity/4.0;
  Control u{each,each,each,each};
  const auto d = dynamics(hover,u,p);
  const auto da = toArray(d);
  for (double value : da) near(value,0.0,2e-13,"hover equilibrium");

  State moving;
  moving.velocity = {3.0,-2.0,1.0};
  const Vec3 drag = bodyDrag(moving,p);
  require(drag.x*moving.velocity.x < 0.0, "x drag direction");
  require(drag.y*moving.velocity.y < 0.0, "y drag direction");
  require(-p.linear_drag.z*moving.velocity.z*moving.velocity.z <= 0.0, "z linear drag direction");

  const Vec3 t0 = rotorTorque({1.0,0.0,0.0,0.0},p);
  const Vec3 t1 = rotorTorque({0.0,1.0,0.0,0.0},p);
  const Vec3 t2 = rotorTorque({0.0,0.0,1.0,0.0},p);
  const Vec3 t3 = rotorTorque({0.0,0.0,0.0,1.0},p);
  require(t0.x>0 && t0.y<0 && t0.z>0, "rotor 1 allocation signs");
  require(t1.x<0 && t1.y<0 && t1.z<0, "rotor 2 allocation signs");
  require(t2.x<0 && t2.y>0 && t2.z>0, "rotor 3 allocation signs");
  require(t3.x>0 && t3.y>0 && t3.z<0, "rotor 4 allocation signs");

  State rotating;
  rotating.body_rate = {2.0,-4.0,6.0};
  const auto qd = dynamics(rotating,{0.0,0.0,0.0,0.0},p).attitude;
  near(qd.w,0.0,1e-14,"quaternion scalar derivative");
  near(qd.x,1.0,1e-14,"quaternion x derivative");
  near(qd.y,-2.0,1e-14,"quaternion y derivative");
  near(qd.z,3.0,1e-14,"quaternion z derivative");
}

void testGeneratedDynamicsAgreement() {
  using namespace quad;
  State state;
  state.position = {1.0,-2.0,3.0};
  state.attitude = normalized(Quaternion{0.82,0.13,-0.31,0.44});
  state.velocity = {4.2,-1.3,0.7};
  state.body_rate = {0.8,-1.1,0.4};
  Control control{2.1,1.7,2.4,1.9};
  const auto x = toArray(state);
  const std::array<double,4> p{1.0,0.0,0.0,0.0};
  const double* inputs[] = {x.data(),control.data(),p.data()};
  std::array<double,kNx> generated{};
  double* outputs[] = {generated.data()};
  int sz_arg=0,sz_res=0,sz_iw=0,sz_w=0;
  quadrotor_expl_ode_fun_work(&sz_arg,&sz_res,&sz_iw,&sz_w);
  std::vector<int> iw(static_cast<std::size_t>(std::max(1,sz_iw)));
  std::vector<double> w(static_cast<std::size_t>(std::max(1,sz_w)));
  const int status = quadrotor_expl_ode_fun(inputs,outputs,iw.data(),w.data(),0);
  require(status==0,"generated dynamics evaluation");
  const auto independent = toArray(dynamics(state,control));
  for (int i=0;i<kNx;++i) near(generated[static_cast<std::size_t>(i)],independent[static_cast<std::size_t>(i)],2e-12,
                                "controller/plant derivative agreement at index "+std::to_string(i));
}

}  // namespace

int main() {
  try {
    testTrajectory(quad::trajectoryConfig(quad::TrajectoryKind::Challenging));
    testTrajectory(quad::trajectoryConfig(quad::TrajectoryKind::DaocpWorst));
    testPaperTrajectoryGrid();
    testTiming();
    testModel();
    testGeneratedDynamicsAgreement();
    std::cout << "All quadrotor unit tests passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Test failure: " << e.what() << '\n';
    return 1;
  }
}
