#include "kvlearn/kvlearn.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <list>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using namespace kvlearn;

// ---------------------------------------------------------------------------
// Synthetic Zipf multimodal trace approximating MM-Session / ShareGPT mix.
// ---------------------------------------------------------------------------
struct TraceItem {
  std::string key;
  int prefix_len = 0;
  Modality modality = Modality::Text;
  double popularity = 0.0;  // unnormalized
};

struct SimMetrics {
  double ttft_ms = 0.0;
  double hit_rate = 0.0;
  double xfer_gib = 0.0;
  uint64_t admits = 0;
  uint64_t discards = 0;
  uint64_t requests = 0;
};

static double zipf_weight(int rank, double alpha) {
  return 1.0 / std::pow(static_cast<double>(rank + 1), alpha);
}

static std::vector<TraceItem> build_catalog(int n_text, int n_image,
                                            int n_video, double alpha) {
  std::vector<TraceItem> items;
  items.reserve(static_cast<size_t>(n_text + n_image + n_video));
  int rank = 0;
  for (int i = 0; i < n_text; ++i, ++rank) {
    TraceItem t;
    t.key = "sys/text" + std::to_string(i);
    t.prefix_len = 64 + (i % 8) * 32;  // 64–288
    t.modality = Modality::Text;
    t.popularity = zipf_weight(rank, alpha) * 0.6;
    items.push_back(t);
  }
  for (int i = 0; i < n_image; ++i, ++rank) {
    TraceItem t;
    t.key = "img/" + std::to_string(i);
    t.prefix_len = 576;  // LLaVA-1.5 visual tokens
    t.modality = Modality::Image;
    t.popularity = zipf_weight(rank, alpha) * 1.4;  // heavier visual reuse
    items.push_back(t);
  }
  for (int i = 0; i < n_video; ++i, ++rank) {
    TraceItem t;
    t.key = "vid/" + std::to_string(i);
    t.prefix_len = 720 + (i % 5) * 180;  // 720–1440
    t.modality = Modality::Video;
    t.popularity = zipf_weight(rank, alpha) * 1.1;
    items.push_back(t);
  }
  return items;
}

enum class Policy { NoCache, LRU, KVLearn };

struct LRUPool {
  struct Node {
    std::string key;
    Bytes footprint = 0;
    int prefix_len = 0;
    Modality modality = Modality::Text;
  };
  Bytes capacity = 0;
  Bytes used = 0;
  std::list<Node> order;
  std::unordered_map<std::string, std::list<Node>::iterator> index;

  explicit LRUPool(Bytes cap) : capacity(cap) {}

  bool hit(const std::string& key) {
    auto it = index.find(key);
    if (it == index.end()) return false;
    order.splice(order.begin(), order, it->second);
    return true;
  }

  void admit(const TraceItem& item, Bytes fp) {
    if (index.count(item.key)) {
      hit(item.key);
      return;
    }
    while (used + fp > capacity && !order.empty()) {
      auto& back = order.back();
      used -= back.footprint;
      index.erase(back.key);
      order.pop_back();
    }
    if (used + fp > capacity) return;
    order.push_front(Node{item.key, fp, item.prefix_len, item.modality});
    index[item.key] = order.begin();
    used += fp;
  }
};

static SimMetrics run_policy(Policy policy, const std::vector<TraceItem>& catalog,
                             const SystemConfig& base_cfg, int n_req,
                             double /*alpha*/, uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<double> weights;
  weights.reserve(catalog.size());
  for (const auto& t : catalog) weights.push_back(t.popularity);
  std::discrete_distribution<size_t> pick(weights.begin(), weights.end());

  SystemConfig cfg = base_cfg;
  CostModel cost(cfg.model, cfg.fabric, cfg.calib, cfg.pool);

  KVLearn kv(cfg);
  kv.set_explore_rate(0.05);
  LRUPool lru(cfg.pool.capacity);

  SimMetrics m;
  double ttft_sum = 0.0;
  Bytes xfer = 0;
  uint64_t hits = 0;

  TimePoint t = 0.0;
  const double interarrival = 0.05;  // 20 req/s

  for (int i = 0; i < n_req; ++i) {
    const TraceItem& item = catalog[pick(rng)];
    t += interarrival;
    const Bytes fp = kv_footprint(cfg.model, item.prefix_len);
    const double R = cost.recompute_ms(item.prefix_len);
    const double T = cost.transfer_ms(fp);

    bool is_hit = false;
    if (policy == Policy::NoCache) {
      is_hit = false;
      xfer += fp;  // recompute + forward
      ttft_sum += R + T;
    } else if (policy == Policy::LRU) {
      is_hit = lru.hit(item.key);
      if (is_hit) {
        ++hits;
        xfer += fp;  // pool → D
        ttft_sum += T;
      } else {
        xfer += fp;          // P → D (or P → pool)
        lru.admit(item, fp);
        xfer += fp;          // admit P → pool
        ttft_sum += R + T;
      }
    } else {
      Request req;
      req.req_id = static_cast<uint64_t>(i);
      req.prefix_key = item.key;
      req.prefix_len = item.prefix_len;
      req.modality = item.modality;
      req.arrival = t;

      auto hid = kv.lookup(req);
      if (hid) {
        is_hit = true;
        ++hits;
        ttft_sum += T;
      } else {
        const BlockId id = static_cast<BlockId>(i + 1);
        auto ar = kv.admit(req, id);
        ttft_sum += R + T;
        (void)ar;
      }
      if (i % 50 == 0) {
        kv.tick(t);
        kv.maybe_emit_horizon_labels(t);
      }
    }
    ++m.requests;
  }

  if (policy == Policy::KVLearn) {
    auto st = kv.stats();
    m.hit_rate = st.hit_rate;
    m.xfer_gib = static_cast<double>(st.transfer_bytes) / (1024.0 * 1024.0 * 1024.0);
    m.admits = st.admits;
    m.discards = st.discards;
  } else {
    m.hit_rate = static_cast<double>(hits) / static_cast<double>(n_req);
    m.xfer_gib = static_cast<double>(xfer) / (1024.0 * 1024.0 * 1024.0);
  }
  m.ttft_ms = ttft_sum / static_cast<double>(n_req);
  return m;
}

static const char* policy_name(Policy p) {
  switch (p) {
    case Policy::NoCache: return "No-Cache";
    case Policy::LRU:     return "LRU-Pool";
    case Policy::KVLearn: return "KVLearn";
  }
  return "?";
}

int main(int argc, char** argv) {
  int n_req = 3000;
  double alpha = 1.1;
  Bytes pool_gb = 64;  // default pool for laptop sim (64 GiB)
  if (argc > 1) n_req = std::atoi(argv[1]);
  if (argc > 2) alpha = std::atof(argv[2]);
  if (argc > 3) pool_gb = std::atoll(argv[3]);

  SystemConfig cfg;
  cfg.model = ModelConfig{32, 8, 128, 2, "LLaMA-3-8B"};
  cfg.fabric.bandwidth_bps = 25e9;  // 200 Gbps
  cfg.pool.capacity = pool_gb * 1024LL * 1024LL * 1024LL;
  cfg.pool.target_util = 0.85;
  cfg.prp.learning_rate = 5e-3;
  cfg.prp.batch_every = 50;
  cfg.prp.horizon_sec = 30.0;  // compact sim horizon
  cfg.atc.Kp = 2e-2;
  cfg.atc.Ki = 5e-4;
  cfg.atc.hit_floor = 0.40;

  auto catalog = build_catalog(/*text*/40, /*image*/80, /*video*/10, alpha);

  std::cout << "KVLearn simulator\n"
            << "  requests=" << n_req << "  zipf_alpha=" << alpha
            << "  pool_GiB=" << pool_gb << "\n"
            << "  model=" << cfg.model.name
            << "  fabric=" << (cfg.fabric.bandwidth_bps * 8 / 1e9) << " Gbps\n\n";

  std::cout << std::left << std::setw(12) << "Method"
            << std::setw(14) << "TTFT(ms)"
            << std::setw(12) << "Hit(%)"
            << std::setw(14) << "Xfer(GiB)"
            << "notes\n";
  std::cout << std::string(60, '-') << "\n";

  for (Policy p : {Policy::NoCache, Policy::LRU, Policy::KVLearn}) {
    auto m = run_policy(p, catalog, cfg, n_req, alpha, /*seed=*/7);
    std::cout << std::left << std::setw(12) << policy_name(p)
              << std::setw(14) << std::fixed << std::setprecision(2) << m.ttft_ms
              << std::setw(12) << std::setprecision(1) << (100.0 * m.hit_rate)
              << std::setw(14) << std::setprecision(3) << m.xfer_gib;
    if (p == Policy::KVLearn) {
      std::cout << "admits=" << m.admits << " discards=" << m.discards;
    }
    std::cout << "\n";
  }

  // Cost-model sanity: P* decreases with L for super-linear prefill (q>1)
  CostModel cost(cfg.model, cfg.fabric, cfg.calib, cfg.pool);
  cost.set_delta_t_est(300.0);
  std::cout << "\nP*_H vs prefix length (Δt=30s, pool=256GiB):\n";
  // Use a large reference pool for the analytic check (not the sim pool size).
  PoolConfig ref_pool;
  ref_pool.capacity = 256LL * 1024 * 1024 * 1024;
  CostModel ref_cost(cfg.model, cfg.fabric, cfg.calib, ref_pool);
  ref_cost.set_delta_t_est(30.0);
  ref_cost.set_avg_recompute_ms(20.0);
  ref_cost.set_arrival_rate(10.0);
  ref_cost.set_storage_pressure(1.0);
  for (int L : {128, 512, 1024, 2048, 4096}) {
    Bytes fp = kv_footprint(cfg.model, L);
    double pstar = ref_cost.optimal_threshold(L, fp, 30.0);
    std::cout << "  L=" << std::setw(5) << L
              << "  R=" << std::setw(8) << std::setprecision(2)
              << ref_cost.recompute_ms(L) << " ms"
              << "  T=" << std::setw(8) << ref_cost.transfer_ms(fp) << " ms"
              << "  P*=" << std::setprecision(4) << pstar << "\n";
  }
  return 0;
}
