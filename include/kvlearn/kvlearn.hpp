#pragma once

#include "kvlearn/atc.hpp"
#include "kvlearn/cars.hpp"
#include "kvlearn/cost_model.hpp"
#include "kvlearn/features.hpp"
#include "kvlearn/pool.hpp"
#include "kvlearn/prp.hpp"
#include "kvlearn/radix_tree.hpp"
#include "kvlearn/types.hpp"

#include <memory>
#include <mutex>

namespace kvlearn {

struct SystemConfig {
  ModelConfig model;
  FabricConfig fabric;
  CostCalib calib;
  PoolConfig pool;
  PRPConfig prp;
  ATCConfig atc;
  FeatureWhitenConfig whiten;
};

// KVLearn control plane (§4): intercepts block-arrive & pool-full events.
class KVLearn {
 public:
  explicit KVLearn(SystemConfig cfg);

  // ---- Critical path (≤ 0.5 ms admission budget) ----

  // Lookup: if prefix resident, record hit and return block id.
  std::optional<BlockId> lookup(const Request& req);

  // Algorithm 1: decide whether to admit freshly computed block into S.
  AdmitResult admit(const Request& req, BlockId new_id);

  // Algorithm 2: free `need` bytes by evicting lowest-CARS blocks.
  EvictResult evict(Bytes need);

  // ---- Offline / feedback path ----
  void on_hit(BlockId id, TimePoint now);
  void maybe_emit_horizon_labels(TimePoint now);
  void tick(TimePoint now);  // ATC + cost EWMA + optional PRP train

  // ---- Introspection ----
  PoolStats stats() const;
  double theta() const { return atc_.theta(); }
  const CostModel& cost_model() const { return cost_; }
  PrefixReusePredictor& prp() { return prp_; }
  KVPool& pool() { return pool_; }
  const KVPool& pool() const { return pool_; }
  RadixTree& radix() { return radix_; }

  // Bootstrap: force-admit a fraction of rejected blocks for label diversity.
  void set_explore_rate(double p) { explore_rate_ = p; }

 private:
  SystemConfig cfg_;
  CostModel cost_;
  FeatureExtractor features_;
  PrefixReusePredictor prp_;
  CarsScorer cars_;
  AdaptiveThresholdController atc_;
  KVPool pool_;
  RadixTree radix_;

  PoolStats stats_{};
  double explore_rate_ = 0.02;  // periodic admission-with-logging
  TimePoint last_tick_ = 0.0;
  double r_bar_ema_ = 20.0;
  double lambda_ema_ = 10.0;
  double dt_evict_ema_ = 30.0;
  TimePoint last_evict_time_ = 0.0;
  uint64_t arrivals_window_ = 0;
  TimePoint window_start_ = 0.0;

  mutable std::mutex mu_;

  PoolStateSnapshot snapshot(Modality m) const;
  void refresh_cars(KVBlock& b, TimePoint now);
  void register_label(KVBlock& b, double y);
  KVBlock make_block(const Request& req, BlockId id) const;
  void update_rates(TimePoint now);
};

}  // namespace kvlearn
