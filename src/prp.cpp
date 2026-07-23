#include "kvlearn/prp.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace kvlearn {

namespace {
inline double sigmoid(double z) {
  if (z >= 0) {
    const double e = std::exp(-z);
    return 1.0 / (1.0 + e);
  }
  const double e = std::exp(z);
  return e / (1.0 + e);
}

inline double relu(double z) { return z > 0.0 ? z : 0.0; }
}  // namespace

PrefixReusePredictor::PrefixReusePredictor(PRPConfig cfg) : cfg_(cfg) {
  init_weights();
}

void PrefixReusePredictor::init_weights() {
  const int h = cfg_.hidden_dim;
  const int d = cfg_.feature_dim;
  W1_.assign(static_cast<size_t>(h * d), 0.0);
  b1_.assign(static_cast<size_t>(h), 0.0);
  w2_.assign(static_cast<size_t>(h), 0.0);
  b2_ = 0.0;

  // Xavier / Glorot uniform
  std::uniform_real_distribution<double> u(-std::sqrt(6.0 / (d + h)),
                                           std::sqrt(6.0 / (d + h)));
  for (double& w : W1_) w = u(rng_);
  std::uniform_real_distribution<double> u2(-std::sqrt(6.0 / (h + 1)),
                                            std::sqrt(6.0 / (h + 1)));
  for (double& w : w2_) w = u2(rng_);
}

double PrefixReusePredictor::forward(const FeatureVec& x,
                                     std::vector<double>* hidden) const {
  const int h = cfg_.hidden_dim;
  const int d = cfg_.feature_dim;
  if (hidden) hidden->assign(static_cast<size_t>(h), 0.0);

  double logit = b2_;
  for (int i = 0; i < h; ++i) {
    double a = b1_[static_cast<size_t>(i)];
    const size_t row = static_cast<size_t>(i * d);
    for (int j = 0; j < d; ++j) {
      a += W1_[row + static_cast<size_t>(j)] * x[static_cast<size_t>(j)];
    }
    a = relu(a);
    if (hidden) (*hidden)[static_cast<size_t>(i)] = a;
    logit += w2_[static_cast<size_t>(i)] * a;
  }
  return sigmoid(logit);
}

double PrefixReusePredictor::predict(const FeatureVec& x) const {
  return forward(x, nullptr);
}

void PrefixReusePredictor::push_label(const FeatureVec& x, double y) {
  Sample s;
  s.x = x;
  s.y = std::clamp(y, 0.0, 1.0);
  if (static_cast<int>(replay_.size()) >= cfg_.replay_capacity) {
    replay_.pop_front();
  }
  replay_.push_back(s);
  ++labels_seen_;
  ++pending_since_train_;
}

void PrefixReusePredictor::sgd_step(const Sample& s) {
  const int h = cfg_.hidden_dim;
  const int d = cfg_.feature_dim;
  std::vector<double> hidden;
  const double p = forward(s.x, &hidden);
  // BCE gradient wrt logit: p − y
  const double dlogit = p - s.y;
  last_loss_ = -(s.y * std::log(std::max(p, 1e-12)) +
                 (1.0 - s.y) * std::log(std::max(1.0 - p, 1e-12)));

  const double lr = cfg_.learning_rate;
  // Output layer
  for (int i = 0; i < h; ++i) {
    const double ha = hidden[static_cast<size_t>(i)];
    // dL/dw2_i = dlogit * h_i
    w2_[static_cast<size_t>(i)] -= lr * dlogit * ha;
  }
  b2_ -= lr * dlogit;

  // Hidden layer (ReLU gate)
  for (int i = 0; i < h; ++i) {
    if (hidden[static_cast<size_t>(i)] <= 0.0) continue;
    const double dh = dlogit * w2_[static_cast<size_t>(i)];
    b1_[static_cast<size_t>(i)] -= lr * dh;
    const size_t row = static_cast<size_t>(i * d);
    for (int j = 0; j < d; ++j) {
      W1_[row + static_cast<size_t>(j)] -=
          lr * dh * s.x[static_cast<size_t>(j)];
    }
  }
  ++train_steps_;
}

int PrefixReusePredictor::train_batch(int max_steps) {
  if (replay_.empty()) return 0;
  std::uniform_int_distribution<size_t> dist(0, replay_.size() - 1);
  const int steps = std::min(max_steps, static_cast<int>(replay_.size()));
  for (int i = 0; i < steps; ++i) {
    sgd_step(replay_[dist(rng_)]);
  }
  return steps;
}

int PrefixReusePredictor::maybe_train() {
  if (pending_since_train_ < static_cast<uint64_t>(cfg_.batch_every)) return 0;
  pending_since_train_ = 0;
  return train_batch(cfg_.batch_every);
}

std::vector<float> PrefixReusePredictor::export_weights() const {
  std::vector<float> out;
  out.reserve(W1_.size() + b1_.size() + w2_.size() + 1);
  for (double v : W1_) out.push_back(static_cast<float>(v));
  for (double v : b1_) out.push_back(static_cast<float>(v));
  for (double v : w2_) out.push_back(static_cast<float>(v));
  out.push_back(static_cast<float>(b2_));
  return out;
}

void PrefixReusePredictor::import_weights(const std::vector<float>& w) {
  const size_t need = W1_.size() + b1_.size() + w2_.size() + 1;
  if (w.size() != need) {
    throw std::invalid_argument("PRP weight size mismatch");
  }
  size_t i = 0;
  for (double& v : W1_) v = w[i++];
  for (double& v : b1_) v = w[i++];
  for (double& v : w2_) v = w[i++];
  b2_ = w[i];
}

}  // namespace kvlearn
