#pragma once

#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <optional>

namespace kvlearn {

// ---------------------------------------------------------------------------
// Basic scalars
// ---------------------------------------------------------------------------
using BlockId = uint64_t;
using TokenId = uint32_t;
using TimePoint = double;   // seconds since epoch / simulation clock
using Bytes = int64_t;

enum class Modality : uint8_t {
  Text = 0,
  Image = 1,
  Video = 2,
};

inline const char* modality_name(Modality m) {
  switch (m) {
    case Modality::Text:  return "text";
    case Modality::Image: return "image";
    case Modality::Video: return "video";
  }
  return "unknown";
}

// ---------------------------------------------------------------------------
// Model / hardware configuration (Eq. 1, Appendix D)
// ---------------------------------------------------------------------------
struct ModelConfig {
  int layers = 32;           // ℓ
  int kv_heads = 8;          // h_kv (GQA)
  int head_dim = 128;        // d_h
  int bytes_per_elem = 2;    // δ (bf16)
  std::string name = "LLaMA-3-8B";
};

struct FabricConfig {
  double bandwidth_bps = 25e9;  // β peak bytes/sec (200 Gbps InfiniBand HDR)
};

struct CostCalib {
  // Piecewise recompute fit (Eq. 8). Defaults follow paper A100 + LLaMA-3-8B
  // magnitudes; runtime uses max(bw, flop) so the curve is monotone and R>T
  // on 200 Gbps for typical prefix lengths (see CostModel::recompute_ms).
  double alpha_bw = 0.026;          // ms / token
  double alpha_flop = 8e-6;         // ms / token²
  double crossover_tokens = 512.0;  // L× (informative; both branches evaluated)
};

struct PoolConfig {
  Bytes capacity = 256LL * 1024 * 1024 * 1024;  // M_S = 256 GB
  double target_util = 0.85;                     // ρ*
  double stale_sec = 30.0;                       // τ_stale
};

struct PRPConfig {
  int feature_dim = 16;          // d
  int hidden_dim = 32;           // h
  double learning_rate = 5e-3;   // η_PRP
  int replay_capacity = 20000;   // |B|
  int batch_every = 100;         // Δn_opt
  double horizon_sec = 600.0;    // H = 10 min reuse horizon
};

struct ATCConfig {
  double Kp = 2e-2;
  double Ki = 5e-4;
  double hit_floor = 0.55;       // h_floor
  double Khr = 0.1;
  double cmax_init = 1e3;        // clamp until observed CARS range grows
};

struct FeatureWhitenConfig {
  int warm_requests = 500;       // n_warm
  double ema = 0.99;             // ϕ
  double eps = 1e-6;
};

// ---------------------------------------------------------------------------
// KV block metadata (control-plane only; tensors live elsewhere)
// ---------------------------------------------------------------------------
struct KVBlock {
  BlockId id = 0;
  std::string prefix_key;        // radix / hash key for the prefix
  int prefix_len = 0;            // L_b tokens
  Modality modality = Modality::Text;
  Bytes footprint = 0;           // |b| bytes
  int tree_depth = 0;
  int num_children = 0;

  TimePoint inserted_at = 0.0;
  TimePoint last_access = 0.0;
  TimePoint first_seen = 0.0;
  uint64_t hit_count = 0;

  // Exponentially decayed access counters (1 / 5 / 15 min windows)
  double freq_1m = 0.0;
  double freq_5m = 0.0;
  double freq_15m = 0.0;

  // Cached scoring state
  double cars = 0.0;
  double p_hat = 0.0;
  TimePoint last_scored = 0.0;
  bool admitted = false;
  bool ever_hit_after_admit = false;
};

// ---------------------------------------------------------------------------
// Request routed through the disaggregated cluster (Appendix C)
// ---------------------------------------------------------------------------
struct Request {
  uint64_t req_id = 0;
  std::string prefix_key;
  int prefix_len = 0;
  Modality modality = Modality::Text;
  int tree_depth = 0;
  int num_children = 0;
  TimePoint arrival = 0.0;
};

enum class AdmitDecision : uint8_t {
  Admit = 0,
  Discard = 1,
  AdmitAfterEvict = 2,
};

struct AdmitResult {
  AdmitDecision decision = AdmitDecision::Discard;
  double cars = 0.0;
  double p_hat = 0.0;
  double theta = 0.0;
  Bytes freed = 0;
};

struct EvictResult {
  std::vector<BlockId> evicted;
  std::vector<KVBlock> evicted_blocks;  // full metadata for PRP labels
  Bytes freed = 0;
};

struct PoolStats {
  Bytes used = 0;
  Bytes capacity = 0;
  double util = 0.0;
  uint64_t num_blocks = 0;
  uint64_t hits = 0;
  uint64_t misses = 0;
  uint64_t admits = 0;
  uint64_t discards = 0;
  uint64_t evictions = 0;
  double hit_rate = 0.0;
  Bytes transfer_bytes = 0;  // cumulative P→pool ingress
};

inline Bytes kv_footprint(const ModelConfig& m, int L) {
  // Eq. (1)/(9): |b| = 2 * ℓ * h_kv * d_h * L * δ
  return static_cast<Bytes>(2LL * m.layers * m.kv_heads * m.head_dim * L *
                            m.bytes_per_elem);
}

inline double now_sec() {
  using clock = std::chrono::steady_clock;
  static const auto t0 = clock::now();
  return std::chrono::duration<double>(clock::now() - t0).count();
}

}  // namespace kvlearn
