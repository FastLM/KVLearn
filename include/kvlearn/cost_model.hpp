#pragma once

#include "kvlearn/types.hpp"

namespace kvlearn {

// Hardware-grounded cost parameters for CARS:
// R(b), T(b)=|b|/β, U(b,Δt)=γ·|b|·Δt, CARS=P̂·(R−T)−U.
class CostModel {
 public:
  CostModel(ModelConfig model, FabricConfig fabric, CostCalib calib,
            PoolConfig pool);

  void set_arrival_rate(double lambda_blocks_per_sec);
  void set_avg_recompute_ms(double r_bar_ms);
  void set_delta_t_est(double dt_sec);
  // Multiply U by pool pressure in [0,1]; empty pool → near-zero opportunity cost.
  void set_storage_pressure(double p);

  // R(b) in milliseconds (piecewise bw / flop fit)
  double recompute_ms(int prefix_len) const;

  // T(b) in milliseconds: |b| / β
  double transfer_ms(Bytes footprint) const;

  // U(b, Δt) in millisecond-equivalent opportunity cost
  // U = γ · |b| · Δt   with γ = R̄ · λ / M_S
  double storage_cost(Bytes footprint, double delta_t_sec) const;

  // Net benefit per hit: R(b) − T(b)  (may be ≤ 0 on slow fabric)
  double net_benefit_ms(int prefix_len, Bytes footprint) const;

  // Break-even reuse threshold: P*_H(b) = U / (R − T)
  double optimal_threshold(int prefix_len, Bytes footprint,
                           double delta_t_sec) const;

  double gamma() const;          // opportunity cost scalar
  double delta_t_est() const { return delta_t_est_; }
  double bandwidth_bps() const { return fabric_.bandwidth_bps; }
  const ModelConfig& model() const { return model_; }
  const PoolConfig& pool() const { return pool_; }

 private:
  ModelConfig model_;
  FabricConfig fabric_;
  CostCalib calib_;
  PoolConfig pool_;

  double lambda_ = 10.0;         // block arrival rate
  double r_bar_ms_ = 20.0;       // running avg recompute
  double delta_t_est_ = 30.0;    // EWMA inter-eviction interval (sec)
  double storage_pressure_ = 0.5;  // updated from pool util
};

}  // namespace kvlearn
