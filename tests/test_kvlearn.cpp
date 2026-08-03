#include "kvlearn/kvlearn.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace kvlearn;

static int g_failed = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " " << #cond      \
                << "\n";                                                       \
      ++g_failed;                                                              \
    }                                                                          \
  } while (0)

static void test_footprint() {
  ModelConfig m{32, 8, 128, 2, "t"};
  // |b| = 2*32*8*128*512*2 = 67,108,864
  CHECK(kv_footprint(m, 512) == 67108864LL);
}

static void test_cost_prop1() {
  SystemConfig cfg;
  cfg.fabric.bandwidth_bps = 25e9;
  // Force super-linear regime for the lengths under test (q>1).
  cfg.calib.alpha_bw = 1e-6;
  cfg.calib.alpha_flop = 8e-6;
  CostModel cost(cfg.model, cfg.fabric, cfg.calib, cfg.pool);
  cost.set_delta_t_est(300.0);
  cost.set_avg_recompute_ms(20.0);
  cost.set_arrival_rate(10.0);

  double prev = 1e9;
  for (int L : {1024, 2048, 4096}) {
    Bytes fp = kv_footprint(cfg.model, L);
    CHECK(cost.recompute_ms(L) > cost.transfer_ms(fp));
    double p = cost.optimal_threshold(L, fp, 300.0);
    CHECK(p < prev);
    prev = p;
  }
}

static void test_prp_learns() {
  PRPConfig cfg;
  cfg.learning_rate = 0.05;
  PrefixReusePredictor prp(cfg);
  FeatureVec pos{};
  FeatureVec neg{};
  pos.fill(0.0);
  neg.fill(0.0);
  // Strongly separable patterns across several dims
  pos[0] = 1.0;
  pos[3] = 1.0;
  pos[6] = 1.0;
  pos[7] = 1.0;
  neg[0] = -1.0;
  neg[3] = 0.0;
  neg[6] = -1.0;
  neg[7] = -1.0;

  for (int epoch = 0; epoch < 20; ++epoch) {
    for (int i = 0; i < 64; ++i) {
      prp.push_label(pos, 1.0);
      prp.push_label(neg, 0.0);
    }
    prp.train_batch(256);
  }

  double p_pos = prp.predict(pos);
  double p_neg = prp.predict(neg);
  CHECK(p_pos > p_neg);
  CHECK(p_pos > 0.6);
  CHECK(p_neg < 0.4);
}

static void test_cars_sign() {
  SystemConfig cfg;
  CostModel cost(cfg.model, cfg.fabric, cfg.calib, cfg.pool);
  cost.set_delta_t_est(60.0);
  CarsScorer cars(&cost);
  Bytes fp = kv_footprint(cfg.model, 1024);
  double high = cars.score(0.9, 1024, fp);
  double low = cars.score(0.05, 1024, fp);
  CHECK(high > low);
}

static void test_atc_raises_on_pressure() {
  AdaptiveThresholdController atc;
  double t0 = atc.theta();
  for (int i = 0; i < 20; ++i) {
    atc.update(/*util=*/0.98, /*hit=*/0.7);
  }
  CHECK(atc.theta() > t0);
}

static void test_pool_evict_order() {
  PoolConfig pc;
  pc.capacity = 1000;
  KVPool pool(pc);

  KVBlock a;
  a.id = 1;
  a.prefix_key = "a";
  a.footprint = 400;
  a.cars = 10.0;
  KVBlock b;
  b.id = 2;
  b.prefix_key = "b";
  b.footprint = 400;
  b.cars = 1.0;  // lower value → evict first
  CHECK(pool.insert(a));
  CHECK(pool.insert(b));

  auto r = pool.evict_bytes(400, nullptr);
  CHECK(r.freed >= 400);
  CHECK(r.evicted.size() == 1);
  CHECK(r.evicted[0] == 2);
  CHECK(pool.contains(1));
  CHECK(!pool.contains(2));
}

static void test_end_to_end_admit() {
  SystemConfig cfg;
  cfg.pool.capacity = 512LL * 1024 * 1024;  // 512 MiB
  cfg.prp.horizon_sec = 30.0;
  KVLearn kv(cfg);

  Request req;
  req.req_id = 1;
  req.prefix_key = "img/42";
  req.prefix_len = 576;
  req.modality = Modality::Image;
  req.arrival = 1.0;

  CHECK(!kv.lookup(req).has_value());
  auto ar = kv.admit(req, /*id=*/1);
  // With cold PRP, admit may or may not pass — either is valid.
  (void)ar;
  kv.tick(1.0);

  // Force-admit by lowering theta and exploring
  kv.set_explore_rate(1.0);
  Request req2 = req;
  req2.req_id = 2;
  req2.prefix_key = "img/43";
  req2.arrival = 2.0;
  auto ar2 = kv.admit(req2, 2);
  CHECK(ar2.decision != AdmitDecision::Discard ||
        kv.pool().size() >= 1);

  auto st = kv.stats();
  CHECK(st.capacity == cfg.pool.capacity);
}

int main() {
  test_footprint();
  test_cost_prop1();
  test_prp_learns();
  test_cars_sign();
  test_atc_raises_on_pressure();
  test_pool_evict_order();
  test_end_to_end_admit();

  if (g_failed) {
    std::cerr << g_failed << " check(s) failed\n";
    return 1;
  }
  std::cout << "All tests passed.\n";
  return 0;
}
