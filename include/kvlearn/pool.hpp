#pragma once

#include "kvlearn/types.hpp"
#include <functional>
#include <optional>
#include <queue>
#include <unordered_map>
#include <vector>

namespace kvlearn {

// Global KV pool directory with CARS min-heap eviction order.
class KVPool {
 public:
  explicit KVPool(PoolConfig cfg);

  bool contains(BlockId id) const;
  bool contains_key(const std::string& prefix_key) const;
  KVBlock* get(BlockId id);
  const KVBlock* get(BlockId id) const;
  KVBlock* get_by_key(const std::string& prefix_key);

  // Insert admitted block. Returns false if footprint exceeds capacity
  // even after full eviction (should not happen if caller evicts first).
  bool insert(KVBlock block);

  // Remove and return block; emits nothing (caller handles PRP labels).
  std::optional<KVBlock> erase(BlockId id);

  // Evict lowest-CARS blocks until `need` bytes freed.
  // `refresh` is called before comparing a candidate (lazy score refresh).
  EvictResult evict_bytes(
      Bytes need,
      const std::function<void(KVBlock&)>& refresh);

  Bytes used() const { return used_; }
  Bytes capacity() const { return cfg_.capacity; }
  double util() const {
    return cfg_.capacity > 0 ? static_cast<double>(used_) / cfg_.capacity : 0.0;
  }
  size_t size() const { return blocks_.size(); }
  const PoolConfig& config() const { return cfg_; }

  // Modality byte accounting for pool-state features.
  Bytes modality_bytes(Modality m) const;
  double modality_fraction(Modality m) const;

  // Iteration (for scoring / stats)
  template <typename Fn>
  void for_each(Fn&& fn) {
    for (auto& kv : blocks_) fn(kv.second);
  }
  template <typename Fn>
  void for_each(Fn&& fn) const {
    for (const auto& kv : blocks_) fn(kv.second);
  }

  void note_eviction(TimePoint now);
  double eviction_rate(TimePoint now) const;

 private:
  struct HeapEntry {
    double cars;
    BlockId id;
    // max-heap inverted → min-CARS via greater comparator in priority_queue
    bool operator>(const HeapEntry& o) const { return cars > o.cars; }
  };

  PoolConfig cfg_;
  std::unordered_map<BlockId, KVBlock> blocks_;
  std::unordered_map<std::string, BlockId> key_index_;
  // Lazy heap: may contain stale entries; validated on pop.
  std::priority_queue<HeapEntry, std::vector<HeapEntry>,
                      std::greater<HeapEntry>>
      heap_;

  Bytes used_ = 0;
  Bytes modality_used_[3] = {0, 0, 0};

  // Eviction-rate window (30s)
  std::vector<TimePoint> recent_evictions_;

  void push_heap(const KVBlock& b);
  void recompute_heap();
};

}  // namespace kvlearn
