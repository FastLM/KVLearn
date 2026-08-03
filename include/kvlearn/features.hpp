#pragma once

#include "kvlearn/types.hpp"
#include <array>

namespace kvlearn {

// d = 16 feature vector (structural 6 + temporal 7 + pool-state 3).
constexpr int kFeatureDim = 16;

using FeatureVec = std::array<double, kFeatureDim>;

struct PoolStateSnapshot {
  double occupancy = 0.0;          // used / capacity
  double modality_frac = 0.0;      // same-modality bytes / used
  double eviction_rate = 0.0;      // blocks/sec over 30s window
};

class FeatureExtractor {
 public:
  explicit FeatureExtractor(FeatureWhitenConfig cfg = {});

  // O(1) feature build + optional whitening.
  FeatureVec extract(const KVBlock& b, const PoolStateSnapshot& pool,
                     TimePoint now) const;

  void observe(const FeatureVec& raw);
  int requests_seen() const { return n_; }

  // Feature index layout:
  //  0 L_b          1 num_children   2 tree_depth
  //  3 is_image     4 is_video       5 log(|b|)
  //  6 freq_1m      7 freq_5m        8 freq_15m
  //  9 log(recency) 10 sin(hour)     11 cos(hour)
  // 12 session_age  13 pool_occ      14 modality_frac
  // 15 eviction_rate
  static FeatureVec raw_features(const KVBlock& b,
                                 const PoolStateSnapshot& pool,
                                 TimePoint now);

 private:
  FeatureWhitenConfig cfg_;
  mutable FeatureVec mean_{};
  mutable FeatureVec m2_{};   // for running variance
  int n_ = 0;

  FeatureVec whiten(const FeatureVec& x) const;
};

// Touch helpers: update exponentially-decayed frequency counters.
void touch_block(KVBlock& b, TimePoint now);

}  // namespace kvlearn
