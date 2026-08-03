#include "kvlearn/cars.hpp"

namespace kvlearn {

CarsScorer::CarsScorer(CostModel* cost) : cost_(cost) {}

double CarsScorer::score(double p_hat, int prefix_len, Bytes footprint) const {
  // CARS(b) = P̂(b)·[R(b)−T(b)] − U(b, Δt_est)
  const double net = cost_->net_benefit_ms(prefix_len, footprint);
  const double u = cost_->storage_cost(footprint, cost_->delta_t_est());
  return p_hat * net - u;
}

double CarsScorer::score_block(const KVBlock& b) const {
  return score(b.p_hat, b.prefix_len, b.footprint);
}

}  // namespace kvlearn
