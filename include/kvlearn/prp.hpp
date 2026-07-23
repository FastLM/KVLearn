#pragma once

#include "kvlearn/features.hpp"
#include "kvlearn/types.hpp"
#include <deque>
#include <random>
#include <vector>

namespace kvlearn {

// Two-layer MLP reuse predictor (§4.2, Eq. 6–7).
// P̂(b) = σ(w₂ᵀ ReLU(W₁ x(b) + b₁) + b₂)
// Params: 32×16 + 32 + 32 + 1 = 577 scalars.
class PrefixReusePredictor {
 public:
  explicit PrefixReusePredictor(PRPConfig cfg = {});

  // Forward pass → reuse probability in [0,1]
  double predict(const FeatureVec& x) const;

  // Push a delayed binary label (hit within H → 1, evicted without hit → 0)
  void push_label(const FeatureVec& x, double y);

  // Run SGD on up to `batch_every` samples from the circular replay buffer.
  // Returns number of gradient steps taken.
  int maybe_train();

  // Force a training step (for tests / offline warm-up).
  int train_batch(int max_steps);

  const PRPConfig& config() const { return cfg_; }
  uint64_t labels_seen() const { return labels_seen_; }
  uint64_t train_steps() const { return train_steps_; }
  double last_loss() const { return last_loss_; }

  // Serialize / restore weights (float32 dump, 577 floats).
  std::vector<float> export_weights() const;
  void import_weights(const std::vector<float>& w);

 private:
  struct Sample {
    FeatureVec x{};
    double y = 0.0;
  };

  PRPConfig cfg_;
  // W1: hidden × feature, stored row-major [h][d]
  std::vector<double> W1_;  // h * d
  std::vector<double> b1_;  // h
  std::vector<double> w2_;  // h
  double b2_ = 0.0;

  std::deque<Sample> replay_;
  uint64_t labels_seen_ = 0;
  uint64_t train_steps_ = 0;
  uint64_t pending_since_train_ = 0;
  double last_loss_ = 0.0;

  mutable std::mt19937 rng_{42};

  void init_weights();
  double forward(const FeatureVec& x, std::vector<double>* hidden) const;
  void sgd_step(const Sample& s);
};

}  // namespace kvlearn
