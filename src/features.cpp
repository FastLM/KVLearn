#include "kvlearn/features.hpp"

#include <algorithm>
#include <cmath>

namespace kvlearn {

namespace {
constexpr double kLogEps = 1e-9;

double safe_log(double x) { return std::log(std::max(x, kLogEps)); }

// Hour-of-day from simulation clock (wrap every 24h).
void hour_sincos(TimePoint now, double* s, double* c) {
  constexpr double kDay = 86400.0;
  constexpr double kPi = 3.14159265358979323846;
  const double hour = std::fmod(std::fmod(now, kDay) + kDay, kDay) / 3600.0;
  const double ang = 2.0 * kPi * hour / 24.0;
  *s = std::sin(ang);
  *c = std::cos(ang);
}
}  // namespace

FeatureExtractor::FeatureExtractor(FeatureWhitenConfig cfg) : cfg_(cfg) {
  mean_.fill(0.0);
  m2_.fill(0.0);
}

FeatureVec FeatureExtractor::raw_features(const KVBlock& b,
                                          const PoolStateSnapshot& pool,
                                          TimePoint now) {
  FeatureVec x{};
  x[0] = static_cast<double>(b.prefix_len);
  x[1] = static_cast<double>(b.num_children);
  x[2] = static_cast<double>(b.tree_depth);
  x[3] = (b.modality == Modality::Image) ? 1.0 : 0.0;
  x[4] = (b.modality == Modality::Video) ? 1.0 : 0.0;
  x[5] = safe_log(static_cast<double>(std::max<Bytes>(b.footprint, 1)));

  x[6] = b.freq_1m;
  x[7] = b.freq_5m;
  x[8] = b.freq_15m;
  x[9] = safe_log(std::max(now - b.last_access, 0.0) + 1.0);

  double hs = 0.0, hc = 0.0;
  hour_sincos(now, &hs, &hc);
  x[10] = hs;
  x[11] = hc;
  x[12] = std::max(now - b.first_seen, 0.0);

  x[13] = pool.occupancy;
  x[14] = pool.modality_frac;
  x[15] = pool.eviction_rate;
  return x;
}

FeatureVec FeatureExtractor::whiten(const FeatureVec& x) const {
  FeatureVec out = x;
  if (n_ < cfg_.warm_requests) return out;
  for (int i = 0; i < kFeatureDim; ++i) {
    const double var = (n_ > 1) ? m2_[i] / static_cast<double>(n_ - 1) : 1.0;
    const double stdv = std::sqrt(std::max(var, 0.0));
    out[i] = (x[i] - mean_[i]) / std::max(stdv, cfg_.eps);
  }
  return out;
}

FeatureVec FeatureExtractor::extract(const KVBlock& b,
                                     const PoolStateSnapshot& pool,
                                     TimePoint now) const {
  return whiten(raw_features(b, pool, now));
}

void FeatureExtractor::observe(const FeatureVec& raw) {
  ++n_;
  // Welford online mean / variance
  for (int i = 0; i < kFeatureDim; ++i) {
    const double delta = raw[i] - mean_[i];
    mean_[i] += delta / static_cast<double>(n_);
    const double delta2 = raw[i] - mean_[i];
    m2_[i] += delta * delta2;
  }
  // After warm-up, blend with EMA for slow drift tracking (ϕ)
  if (n_ > cfg_.warm_requests) {
    for (int i = 0; i < kFeatureDim; ++i) {
      mean_[i] = cfg_.ema * mean_[i] + (1.0 - cfg_.ema) * raw[i];
    }
  }
}

void touch_block(KVBlock& b, TimePoint now) {
  // Exponential decay of frequency counters toward windows of 60/300/900s.
  const double dt = std::max(now - b.last_access, 0.0);
  auto decay = [](double v, double dt, double tau) {
    return v * std::exp(-dt / tau);
  };
  b.freq_1m = decay(b.freq_1m, dt, 60.0) + 1.0;
  b.freq_5m = decay(b.freq_5m, dt, 300.0) + 1.0;
  b.freq_15m = decay(b.freq_15m, dt, 900.0) + 1.0;
  b.last_access = now;
  ++b.hit_count;
}

}  // namespace kvlearn
