#include "kvlearn/pool.hpp"

#include <algorithm>

namespace kvlearn {

KVPool::KVPool(PoolConfig cfg) : cfg_(cfg) {}

bool KVPool::contains(BlockId id) const {
  return blocks_.find(id) != blocks_.end();
}

bool KVPool::contains_key(const std::string& prefix_key) const {
  return key_index_.find(prefix_key) != key_index_.end();
}

KVBlock* KVPool::get(BlockId id) {
  auto it = blocks_.find(id);
  return it == blocks_.end() ? nullptr : &it->second;
}

const KVBlock* KVPool::get(BlockId id) const {
  auto it = blocks_.find(id);
  return it == blocks_.end() ? nullptr : &it->second;
}

KVBlock* KVPool::get_by_key(const std::string& prefix_key) {
  auto it = key_index_.find(prefix_key);
  if (it == key_index_.end()) return nullptr;
  return get(it->second);
}

void KVPool::push_heap(const KVBlock& b) {
  heap_.push(HeapEntry{b.cars, b.id});
}

bool KVPool::insert(KVBlock block) {
  if (block.footprint <= 0) return false;
  if (used_ + block.footprint > cfg_.capacity) return false;
  if (blocks_.count(block.id)) return false;

  used_ += block.footprint;
  const int mi = static_cast<int>(block.modality);
  if (mi >= 0 && mi < 3) modality_used_[mi] += block.footprint;

  key_index_[block.prefix_key] = block.id;
  push_heap(block);
  blocks_.emplace(block.id, std::move(block));
  return true;
}

std::optional<KVBlock> KVPool::erase(BlockId id) {
  auto it = blocks_.find(id);
  if (it == blocks_.end()) return std::nullopt;

  KVBlock b = std::move(it->second);
  used_ -= b.footprint;
  const int mi = static_cast<int>(b.modality);
  if (mi >= 0 && mi < 3) modality_used_[mi] -= b.footprint;
  key_index_.erase(b.prefix_key);
  blocks_.erase(it);
  // Lazy heap: stale entries cleaned on pop.
  return b;
}

EvictResult KVPool::evict_bytes(
    Bytes need, const std::function<void(KVBlock&)>& refresh) {
  EvictResult out;
  if (need <= 0) return out;

  // Refresh bottom candidates once, then drain min-heap with a safety cap.
  std::vector<BlockId> ids;
  ids.reserve(blocks_.size());
  for (auto& kv : blocks_) ids.push_back(kv.first);

  for (BlockId id : ids) {
    auto* b = get(id);
    if (!b) continue;
    if (refresh) refresh(*b);
  }
  recompute_heap();

  const size_t safety = blocks_.size() * 4 + 8;
  size_t steps = 0;
  while (out.freed < need && !heap_.empty() && steps++ < safety) {
    HeapEntry top = heap_.top();
    heap_.pop();
    auto* live = get(top.id);
    if (!live) continue;  // stale
    if (std::abs(live->cars - top.cars) > 1e-6) {
      push_heap(*live);
      continue;
    }
    auto removed = erase(top.id);
    if (!removed) continue;
    out.evicted.push_back(removed->id);
    out.freed += removed->footprint;
    out.evicted_blocks.push_back(std::move(*removed));
  }
  return out;
}

void KVPool::recompute_heap() {
  while (!heap_.empty()) heap_.pop();
  for (const auto& kv : blocks_) push_heap(kv.second);
}

Bytes KVPool::modality_bytes(Modality m) const {
  const int mi = static_cast<int>(m);
  if (mi < 0 || mi >= 3) return 0;
  return modality_used_[mi];
}

double KVPool::modality_fraction(Modality m) const {
  if (used_ <= 0) return 0.0;
  return static_cast<double>(modality_bytes(m)) / static_cast<double>(used_);
}

void KVPool::note_eviction(TimePoint now) {
  recent_evictions_.push_back(now);
  // Drop events older than 30s
  const TimePoint cut = now - 30.0;
  recent_evictions_.erase(
      std::remove_if(recent_evictions_.begin(), recent_evictions_.end(),
                     [cut](TimePoint t) { return t < cut; }),
      recent_evictions_.end());
}

double KVPool::eviction_rate(TimePoint now) const {
  int n = 0;
  const TimePoint cut = now - 30.0;
  for (TimePoint t : recent_evictions_) {
    if (t >= cut) ++n;
  }
  return static_cast<double>(n) / 30.0;
}

}  // namespace kvlearn
