#pragma once

#include "kvlearn/types.hpp"

namespace kvlearn {

// Adaptive Threshold Controller:
//   e(t) = ρ_pool(t) − ρ*
//   θ(t+1) = θ(t) + Kp·e(t) + Ki·Σ e(j)
// plus hit-rate secondary bias Δθ_hr.
class AdaptiveThresholdController {
 public:
  explicit AdaptiveThresholdController(ATCConfig cfg = {},
                                       double rho_star = 0.85);

  // Update from latest pool util + observed hit rate; returns new θ.
  double update(double pool_util, double hit_rate);

  double theta() const { return theta_; }
  void set_theta(double t) { theta_ = t; }
  void set_cmax(double c) { cmax_ = std::max(c, 1e-6); }
  double integral() const { return integral_; }
  const ATCConfig& config() const { return cfg_; }

 private:
  ATCConfig cfg_;
  double rho_star_;
  double theta_ = 0.0;
  double integral_ = 0.0;
  double cmax_;

  double clamp(double t) const;
};

}  // namespace kvlearn
