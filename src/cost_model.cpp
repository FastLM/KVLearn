#include "kvlearn/cost_model.hpp"

#include <algorithm>
#include <cmath>

namespace kvlearn {

CostModel::CostModel(ModelConfig model, FabricConfig fabric, CostCalib calib,
                     PoolConfig pool)
    : model_(std::move(model)),
      fabric_(fabric),
      calib_(calib),
      pool_(pool) {}

void CostModel::set_arrival_rate(double lambda_blocks_per_sec) {
  lambda_ = std::max(1e-6, lambda_blocks_per_sec);
}

void CostModel::set_avg_recompute_ms(double r_bar_ms) {
  r_bar_ms_ = std::max(1e-6, r_bar_ms);
}

void CostModel::set_delta_t_est(double dt_sec) {
  delta_t_est_ = std::max(1e-3, dt_sec);
}

void CostModel::set_storage_pressure(double p) {
  storage_pressure_ = std::clamp(p, 0.0, 1.0);
}

double CostModel::recompute_ms(int prefix_len) const {
  const double L = static_cast<double>(std::max(prefix_len, 1));
  const double r_bw = calib_.alpha_bw * L;
  const double r_flop = calib_.alpha_flop * L * L;
  // Monotone envelope of the two regimes in Eq. (8).
  (void)calib_.crossover_tokens;
  return std::max(r_bw, r_flop);
}

double CostModel::transfer_ms(Bytes footprint) const {
  if (fabric_.bandwidth_bps <= 0.0) return 1e9;
  // bytes / (bytes/sec) → sec → ms
  return (static_cast<double>(footprint) / fabric_.bandwidth_bps) * 1e3;
}

double CostModel::gamma() const {
  // γ = R̄ · λ / M_S   (per-byte-per-second opportunity cost, in ms units)
  if (pool_.capacity <= 0) return 0.0;
  return (r_bar_ms_ * lambda_) / static_cast<double>(pool_.capacity);
}

double CostModel::storage_cost(Bytes footprint, double delta_t_sec) const {
  // Opportunity cost scales with how full the pool is (empty → U≈0).
  return gamma() * static_cast<double>(footprint) * delta_t_sec *
         storage_pressure_;
}

double CostModel::net_benefit_ms(int prefix_len, Bytes footprint) const {
  return recompute_ms(prefix_len) - transfer_ms(footprint);
}

double CostModel::optimal_threshold(int prefix_len, Bytes footprint,
                                    double delta_t_sec) const {
  const double denom = net_benefit_ms(prefix_len, footprint);
  if (denom <= 1e-12) return 1.0;  // never worth keeping if R ≤ T
  // May exceed 1 when U dominates (R−T): interpret as "never admit".
  return storage_cost(footprint, delta_t_sec) / denom;
}

}  // namespace kvlearn
