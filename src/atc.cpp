#include "kvlearn/atc.hpp"

#include <algorithm>
#include <cmath>

namespace kvlearn {

AdaptiveThresholdController::AdaptiveThresholdController(ATCConfig cfg,
                                                         double rho_star)
    : cfg_(cfg), rho_star_(rho_star), cmax_(cfg.cmax_init) {}

double AdaptiveThresholdController::clamp(double t) const {
  return std::clamp(t, -cmax_, cmax_);
}

double AdaptiveThresholdController::update(double pool_util, double hit_rate) {
  // Primary PI on pool pressure
  const double e = pool_util - rho_star_;
  integral_ += e;
  // Anti-windup: freeze integral when saturated against error sign
  theta_ = theta_ + cfg_.Kp * e + cfg_.Ki * integral_;

  // Secondary hit-rate bias: raise θ when hit rate falls below floor
  if (hit_rate < cfg_.hit_floor) {
    theta_ += cfg_.Khr * (cfg_.hit_floor - hit_rate);
  }

  theta_ = clamp(theta_);

  // Soft anti-windup
  if ((theta_ >= cmax_ && e > 0) || (theta_ <= -cmax_ && e < 0)) {
    integral_ -= e;
  }
  return theta_;
}

}  // namespace kvlearn
