#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace quad {

constexpr double kPi = 3.141592653589793238462643383279502884;

struct Vec3 {
  double x{};
  double y{};
  double z{};

  double& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
  const double& operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
};

inline Vec3 operator+(Vec3 a, const Vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator-(const Vec3& a) { return {-a.x, -a.y, -a.z}; }
inline Vec3 operator*(const Vec3& a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3 operator*(double s, const Vec3& a) { return a * s; }
inline Vec3 operator/(const Vec3& a, double s) { return {a.x / s, a.y / s, a.z / s}; }
inline Vec3& operator+=(Vec3& a, const Vec3& b) { a = a + b; return a; }

inline double dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
inline double squaredNorm(const Vec3& a) { return dot(a, a); }
inline double norm(const Vec3& a) { return std::sqrt(squaredNorm(a)); }
inline Vec3 normalized(const Vec3& a) {
  const double n = norm(a);
  if (!(n > 1e-14)) throw std::runtime_error("cannot normalize near-zero vector");
  return a / n;
}

inline Vec3 normalizedDerivative(const Vec3& a, const Vec3& adot) {
  const double n = norm(a);
  if (!(n > 1e-14)) throw std::runtime_error("singular normalized-vector derivative");
  const Vec3 ahat = a / n;
  return (adot - ahat * dot(ahat, adot)) / n;
}

struct Quaternion {
  double w{1.0};
  double x{};
  double y{};
  double z{};
};

inline Quaternion operator-(const Quaternion& q) { return {-q.w, -q.x, -q.y, -q.z}; }
inline Quaternion operator+(const Quaternion& a, const Quaternion& b) {
  return {a.w+b.w, a.x+b.x, a.y+b.y, a.z+b.z};
}
inline Quaternion operator*(const Quaternion& q, double s) { return {q.w*s, q.x*s, q.y*s, q.z*s}; }
inline Quaternion multiply(const Quaternion& a, const Quaternion& b) {
  return {
    a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
    a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
    a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
    a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w};
}
inline Quaternion conjugate(const Quaternion& q) { return {q.w, -q.x, -q.y, -q.z}; }
inline double dot(const Quaternion& a, const Quaternion& b) {
  return a.w*b.w + a.x*b.x + a.y*b.y + a.z*b.z;
}
inline double norm(const Quaternion& q) { return std::sqrt(dot(q, q)); }
inline Quaternion normalized(const Quaternion& q) {
  const double n = norm(q);
  if (!(n > 1e-14)) throw std::runtime_error("cannot normalize near-zero quaternion");
  return q * (1.0/n);
}

struct Mat3 {
  // Body axes expressed in the inertial frame (column-major rotation matrix).
  Vec3 x{1.0, 0.0, 0.0};
  Vec3 y{0.0, 1.0, 0.0};
  Vec3 z{0.0, 0.0, 1.0};
};

inline Vec3 operator*(const Mat3& r, const Vec3& v) {
  return r.x*v.x + r.y*v.y + r.z*v.z;
}
inline Vec3 transposeMultiply(const Mat3& r, const Vec3& v) {
  return {dot(r.x, v), dot(r.y, v), dot(r.z, v)};
}

Mat3 rotationMatrix(const Quaternion& q);
Quaternion quaternionFromRotation(const Mat3& r);

inline double clamp(double x, double lo, double hi) { return std::max(lo, std::min(hi, x)); }

}  // namespace quad

