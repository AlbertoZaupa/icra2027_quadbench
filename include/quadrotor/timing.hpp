#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace quad {

inline double average(const std::vector<double>& values) {
  return values.empty() ? std::numeric_limits<double>::quiet_NaN()
    : std::accumulate(values.begin(), values.end(), 0.0)/static_cast<double>(values.size());
}

inline double maximum(const std::vector<double>& values) {
  return values.empty() ? std::numeric_limits<double>::quiet_NaN()
    : *std::max_element(values.begin(), values.end());
}

inline std::vector<double> perSamplingAverages(const std::vector<double>& values,
                                              const std::vector<int>& sampling_index) {
  if (values.size() != sampling_index.size())
    throw std::runtime_error("timing values and sampling indices have different sizes");
  std::map<int, std::pair<double, std::uint64_t>> grouped;
  for (std::size_t i = 0; i < values.size(); ++i) {
    auto& entry = grouped[sampling_index[i]];
    entry.first += values[i];
    ++entry.second;
  }
  std::vector<double> result;
  result.reserve(grouped.size());
  for (const auto& item : grouped)
    result.push_back(item.second.first/static_cast<double>(item.second.second));
  return result;
}

// Fractions sum to one, including when several solvers tie exactly.
inline std::vector<double> fastestShares(const std::vector<double>& times) {
  if (times.empty()) throw std::invalid_argument("empty solver timing selection");
  const double best = *std::min_element(times.begin(), times.end());
  const auto ties = std::count(times.begin(), times.end(), best);
  std::vector<double> shares(times.size(), 0.0);
  for (std::size_t i = 0; i < times.size(); ++i)
    if (times[i] == best) shares[i] = 1.0/static_cast<double>(ties);
  return shares;
}

}  // namespace quad
