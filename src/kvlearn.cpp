#include "kvlearn/kvlearn.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace kvlearn {

KVLearn::KVLearn(SystemConfig cfg)
    : cfg_(std::move(cfg)),
      cost_(cfg_.model, cfg_.fabric, cfg_.calib, cfg_.pool),
      features_(cfg_.whiten),
      prp_(cfg_.prp),
      cars_(&cost_),
      atc_(cfg_.atc, cfg_.pool.target_util),
      pool_(cfg_.pool) {
  stats_.capacity = cfg_.pool.capacity;
}

PoolStateSnapshot KVLearn::snapshot(Modality m) const {
  PoolStateSnapshot s;
  s.occupancy = pool_.util();
  s.modality_frac = pool_.modality_fraction(m);
  s.eviction_rate = pool_.eviction_rate(last_tick_);
  return s;
}

KVBlock KVLearn::make_block(const Request& req, BlockId id) const {
  KVBlock b;
  b.id = id;
  b.prefix_key = req.prefix_key;
  b.prefix_len = req.prefix_len;
  b.modality = req.modality;
  b.footprint = kv_footprint(cfg_.model, req.prefix_len);
  b.tree_depth = req.tree_depth;
  b.num_children = req.num_children;
  b.inserted_at = req.arrival;
  b.last_access = req.arrival;
  b.first_seen = req.arrival;
  b.freq_1m = 1.0;
  b.freq_5m = 1.0;
  b.freq_15m = 1.0;
  return b;
}

void KVLearn::refresh_cars(KVBlock& b, TimePoint now) {
  if (now - b.last_scored < cfg_.pool.stale_sec && b.last_scored > 0.0) {
    return;
  }
  cost_.set_storage_pressure(pool_.util());
  const auto snap = snapshot(b.modality);
  const FeatureVec x = features_.extract(b, snap, now);
  b.p_hat = prp_.predict(x);
  b.cars = cars_.score(b.p_hat, b.prefix_len, b.footprint);
  b.last_scored = now;
  atc_.set_cmax(std::max(atc_.config().cmax_init, std::abs(b.cars) * 2.0));
}

void KVLearn::register_label(KVBlock& b, double y) {
  const auto snap = snapshot(b.modality);
  const FeatureVec raw =
      FeatureExtractor::raw_features(b, snap, b.last_access);
  features_.observe(raw);
  const FeatureVec x = features_.extract(b, snap, b.last_access);
  prp_.push_label(x, y);
}

std::optional<BlockId> KVLearn::lookup(const Request& req) {
  std::lock_guard<std::mutex> lock(mu_);
  ++arrivals_window_;
  update_rates(req.arrival);

  auto* b = pool_.get_by_key(req.prefix_key);
  if (!b) {
    ++stats_.misses;
    stats_.hit_rate =
        static_cast<double>(stats_.hits) /
        static_cast<double>(std::max<uint64_t>(1, stats_.hits + stats_.misses));
    return std::nullopt;
  }

  touch_block(*b, req.arrival);
  b->ever_hit_after_admit = true;
  ++stats_.hits;
  stats_.hit_rate =
      static_cast<double>(stats_.hits) /
      static_cast<double>(std::max<uint64_t>(1, stats_.hits + stats_.misses));

  // Hit still pays pool→D transfer (counted for metrics).
  stats_.transfer_bytes += b->footprint;
  return b->id;
}

AdmitResult KVLearn::admit(const Request& req, BlockId new_id) {
  std::lock_guard<std::mutex> lock(mu_);
  AdmitResult res;
  res.theta = atc_.theta();

  KVBlock b = make_block(req, new_id);
  auto path = radix_.upsert(req.prefix_key, new_id);
  b.tree_depth = path.depth;
  b.num_children = path.num_children;

  const auto snap = snapshot(b.modality);
  const FeatureVec raw = FeatureExtractor::raw_features(b, snap, req.arrival);
  features_.observe(raw);
  const FeatureVec x = features_.extract(b, snap, req.arrival);

  b.p_hat = prp_.predict(x);
  b.cars = cars_.score(b.p_hat, b.prefix_len, b.footprint);
  b.last_scored = req.arrival;
  res.p_hat = b.p_hat;
  res.cars = b.cars;

  // Track recompute EMA for γ
  const double r = cost_.recompute_ms(b.prefix_len);
  r_bar_ema_ = 0.95 * r_bar_ema_ + 0.05 * r;
  cost_.set_avg_recompute_ms(r_bar_ema_);
  cost_.set_storage_pressure(pool_.util());

  const bool pass = b.cars > atc_.theta();

  // Exploration: occasionally admit rejected blocks for label bootstrap (§4.2)
  static thread_local std::mt19937 rng{123};
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const bool explore = !pass && (u(rng) < explore_rate_);

  if (!pass && !explore) {
    ++stats_.discards;
    res.decision = AdmitDecision::Discard;
    // Miss path still transfers P→D once.
    stats_.transfer_bytes += b.footprint;
    return res;
  }

  // Need room?
  Bytes need = 0;
  if (pool_.used() + b.footprint > pool_.capacity()) {
    need = pool_.used() + b.footprint - pool_.capacity();
  }

  if (need > 0) {
    auto freed = pool_.evict_bytes(need, [this, t = req.arrival](KVBlock& blk) {
      this->refresh_cars(blk, t);
    });
    res.freed = freed.freed;
    for (auto& victim : freed.evicted_blocks) {
      // Alg. 2: emit y=0 if never reused before eviction; y=1 if it had hits.
      const double y = (victim.hit_count > 1) ? 1.0 : 0.0;
      register_label(victim, y);
      radix_.clear_block(victim.prefix_key);
      pool_.note_eviction(req.arrival);
      ++stats_.evictions;
      if (last_evict_time_ > 0.0) {
        const double gap = req.arrival - last_evict_time_;
        dt_evict_ema_ = 0.9 * dt_evict_ema_ + 0.1 * std::max(gap, 1e-3);
        cost_.set_delta_t_est(dt_evict_ema_);
      }
      last_evict_time_ = req.arrival;
    }

    // Re-check capacity after eviction
    if (pool_.used() + b.footprint > pool_.capacity()) {
      ++stats_.discards;
      res.decision = AdmitDecision::Discard;
      stats_.transfer_bytes += b.footprint;
      return res;
    }
    res.decision = AdmitDecision::AdmitAfterEvict;
  } else {
    res.decision = AdmitDecision::Admit;
  }

  b.admitted = true;
  // P→pool admission transfer
  stats_.transfer_bytes += b.footprint;
  pool_.insert(std::move(b));
  ++stats_.admits;
  stats_.used = pool_.used();
  stats_.num_blocks = pool_.size();
  stats_.util = pool_.util();
  return res;
}

EvictResult KVLearn::evict(Bytes need) {
  std::lock_guard<std::mutex> lock(mu_);
  TimePoint now = last_tick_ > 0 ? last_tick_ : now_sec();

  auto result = pool_.evict_bytes(need, [this, now](KVBlock& blk) {
    this->refresh_cars(blk, now);
  });

  for (auto& victim : result.evicted_blocks) {
    // Alg. 2 line 7: y=0 for eviction without reuse signal preference;
    // if the block was hit, still a useful (positive) training signal.
    const double y = (victim.hit_count > 1) ? 1.0 : 0.0;
    register_label(victim, y);
    radix_.clear_block(victim.prefix_key);
    pool_.note_eviction(now);
    ++stats_.evictions;
    if (last_evict_time_ > 0.0) {
      dt_evict_ema_ =
          0.9 * dt_evict_ema_ + 0.1 * std::max(now - last_evict_time_, 1e-3);
      cost_.set_delta_t_est(dt_evict_ema_);
    }
    last_evict_time_ = now;
  }

  stats_.used = pool_.used();
  stats_.num_blocks = pool_.size();
  stats_.util = pool_.util();
  return result;
}

void KVLearn::on_hit(BlockId id, TimePoint now) {
  std::lock_guard<std::mutex> lock(mu_);
  auto* b = pool_.get(id);
  if (!b) return;
  touch_block(*b, now);
  b->ever_hit_after_admit = true;
}

void KVLearn::maybe_emit_horizon_labels(TimePoint now) {
  std::lock_guard<std::mutex> lock(mu_);
  const double H = cfg_.prp.horizon_sec;
  std::vector<BlockId> to_label;
  pool_.for_each([&](KVBlock& b) {
    if (!b.admitted) return;
    // Positive label when reused within horizon after admit
    if (b.ever_hit_after_admit && b.hit_count >= 2) {
      register_label(b, 1.0);
      b.ever_hit_after_admit = false;  // one-shot per reuse episode
    } else if ((now - b.inserted_at) > H && b.hit_count <= 1) {
      // Never reused within horizon → negative
      register_label(b, 0.0);
      to_label.push_back(b.id);
    }
  });
  // Optionally drop stale never-hit blocks (policy choice)
  for (BlockId id : to_label) {
    auto gone = pool_.erase(id);
    if (gone) {
      radix_.clear_block(gone->prefix_key);
      ++stats_.evictions;
    }
  }
  prp_.maybe_train();
}

void KVLearn::update_rates(TimePoint now) {
  if (window_start_ <= 0.0) window_start_ = now;
  const double dt = now - window_start_;
  if (dt >= 1.0) {
    const double rate = static_cast<double>(arrivals_window_) / dt;
    lambda_ema_ = 0.9 * lambda_ema_ + 0.1 * rate;
    cost_.set_arrival_rate(lambda_ema_);
    arrivals_window_ = 0;
    window_start_ = now;
  }
}

void KVLearn::tick(TimePoint now) {
  std::lock_guard<std::mutex> lock(mu_);
  last_tick_ = now;
  update_rates(now);

  stats_.used = pool_.used();
  stats_.num_blocks = pool_.size();
  stats_.util = pool_.util();
  stats_.capacity = pool_.capacity();

  atc_.update(pool_.util(), stats_.hit_rate);
  prp_.maybe_train();
}

PoolStats KVLearn::stats() const {
  std::lock_guard<std::mutex> lock(mu_);
  return stats_;
}

}  // namespace kvlearn
