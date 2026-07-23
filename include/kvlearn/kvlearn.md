# Namespace API

```cpp
#include "kvlearn/kvlearn.hpp"

kvlearn::SystemConfig cfg;
cfg.pool.capacity = 256LL << 30;          // 256 GiB
cfg.fabric.bandwidth_bps = 25e9;          // 200 Gbps
cfg.model = {32, 8, 128, 2, "LLaMA-3-8B"};

kvlearn::KVLearn ctrl(cfg);

kvlearn::Request req;
req.prefix_key = "img/product_42";
req.prefix_len = 576;
req.modality = kvlearn::Modality::Image;
req.arrival = kvlearn::now_sec();

if (auto id = ctrl.lookup(req)) {
  // pool hit → schedule S → D transfer
} else {
  // miss → prefill on P-node, then:
  auto decision = ctrl.admit(req, /*new_block_id=*/42);
  if (decision.decision == kvlearn::AdmitDecision::Discard) {
    // forward KV only to D-node; do not replicate into S
  }
}

ctrl.tick(kvlearn::now_sec());
ctrl.maybe_emit_horizon_labels(kvlearn::now_sec());
```
