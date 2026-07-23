#pragma once

#include "kvlearn/cost_model.hpp"
#include "kvlearn/types.hpp"

namespace kvlearn {

// Cost-Aware Retention Score (§4.3, Eq. 10):
//   CARS(b) = P̂(b) · [R(b) − T(b)] − U(b, Δt_est)
class CarsScorer {
 public:
  explicit CarsScorer(CostModel* cost);

  double score(double p_hat, int prefix_len, Bytes footprint) const;

  // Convenience: score a resident / candidate block.
  double score_block(const KVBlock& b) const;

 private:
  CostModel* cost_;
};

}  // namespace kvlearn
