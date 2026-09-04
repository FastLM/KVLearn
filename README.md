# KVLearn: Learning KV Cache Retention in Disaggregated LLM Serving Systems

| Component | Role |
|-----------|------|
| **PRP** | Prefix Reuse Predictor — `P̂(b) = f_θ(x(b)) ∈ [0,1]`, updated online from delayed reuse labels |
| **CARS** | Cost-Aware Retention Score — `CARS(b) = P̂(b)·(R(b)−T(b)) − U(b,Δt)` |
| **ATC** | Adaptive Threshold Controller — adapts `θ` from pool pressure and hit rate |

Admission and eviction run in the KV-pool coordinator path.

## Build

**Make:**

```bash
make -j
make test
./build/kvlearn_sim [n_requests] [zipf_alpha] [pool_GiB]
```

**CMake:**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Example:

```bash
./build/kvlearn_sim 3000 1.1 4
```

## Layout

```
include/kvlearn/   public headers (types, cost model, PRP, CARS, ATC, pool, control plane)
src/               implementations
apps/simulate.cpp  Zipf multimodal trace vs No-Cache / LRU-Pool / KVLearn
tests/             unit checks (threshold scaling, PRP learning, heap eviction, …)
```

## Cost model

```text
|b|   = 2 · ℓ · h_kv · d_h · L · δ
R(b)  ≈ α_bw·L          if L < L×   (bandwidth-bound)
      ≈ α_flop·L²       if L ≥ L×   (compute-bound)
T(b)  = |b| / β
U(b)  = γ · |b| · Δt    γ = R̄·λ / M_S
CARS  = P̂(b)·(R−T) − U
KEEP  ⇔  CARS > θ       (θ from ATC)
```

For super-linear prefill (`q>1`), the optimal reuse threshold `P*_H` **decreases** with prefix length — longer visual blocks admit at lower predicted reuse.

## Integrating into a serving stack

1. Co-locate `kvlearn::KVLearn` with the global KV pool coordinator (Mooncake-style `P → S → D`).
2. On radix hit → `lookup(req)`; on miss after prefill → `admit(req, block_id)`.
3. Feed delayed labels via eviction (`y=0/1`) and `maybe_emit_horizon_labels`.
4. Call `tick(now)` periodically so ATC tracks `ρ_pool` and PRP trains off-path.

Tensor movement (RDMA / NCCL) stays in your data plane; this library only decides **keep vs discard**.

## Defaults settings

- Features `d=16`, hidden `h=32`, `η=5e−3`, replay `|B|=20k`, batch every 100 labels
- ATC: `ρ*=0.85`, `Kp=2e−2`, `Ki=5e−4`, `h_floor=0.55`
- A100 + LLaMA-3-8B calib: `α_bw≈0.026 ms/tok`, `α_flop≈8e−6 ms/tok²`, `L×=512`
- Fabric default: InfiniBand HDR 200 Gbps (`β=25 GB/s`)

## Citation

If you use this code of KVLearn, please cite:

```bibtex
@inproceedings{kvlearn,
  author = {Liu, Dong and Yu, Yanxuan and Jiang, Eric and Wang, Shu and Wu, Ying Nian},
  title = {To Keep or Not to Keep: Learning KV Cache Retention in Disaggregated LLM Serving Systems},
  year = {2026},
  isbn = {9798400724732},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3793230.3837769},
  doi = {10.1145/3793230.3837769},
  abstract = {Disaggregated LLM serving separates prefill and decode into distinct node pools, interposing a network fabric between the moment a key-value (KV) cache is computed and the moment it is consumed. This architectural shift invalidates a core assumption of classical cache policies: that the cost of a miss is simply recomputation on the same device. In disaggregated systems, a miss triggers both recomputation on a prefill node and a network transfer of the resulting KV block to the decode node—costs that differ by an order of magnitude and depend on prefix length, model width, and fabric bandwidth. Meanwhile, admitting a block to the global KV pool requires an additional transfer at compute time, so a poorly chosen keep decision wastes both memory and bandwidth even before reuse occurs.We present KVLearn, a learning-based retention framework that makes keep/evict decisions as first-class cost-optimization choices in disaggregated LLM serving. KVLearn consists of three components: (i) a lightweight Prefix Reuse Predictor (PRP) that estimates reuse probability from structural and temporal prefix features without touching model weights; (ii) a Cost-Aware Retention Score (CARS) that translates reuse probability into a keep/admit signal by accounting for per-block recompute, transfer, and storage costs; and (iii) an Adaptive Threshold Controller (ATC) that adjusts the admission threshold online using closed-loop feedback from observed hit rates and memory pressure. We integrate KVLearn into a globally disaggregated serving topology and evaluate it on both text and multimodal workloads, where image/video-derived tokens create large, expensive-to-recompute KV blocks under heterogeneous reuse distributions. KVLearn reduces end-to-end time-to-first-token (TTFT) by up to 56\% vs. No-Cache (recompute-only), up to 38\% vs. LRU-Pool, and up to 33\% vs. Mooncake-style disaggregated baselines. Inter-node KV transfer volume is cut by up to 53\% vs. LRU-Pool. On MM-Session, throughput stays within ~5\% of oracle. Our code implementation of KVLearn is available at https://github.com/FastLM/KVLearn.},
  booktitle = {Proceedings of the 19th ACM International Systems and Storage Conference},
  pages = {52–65},
  numpages = {14},
  keywords = {KV cache, disaggregated LLM serving, cache admission, online learning, efficient inference, memory hierarchy, prefix reuse},
  location = {Virtual Event, Virtual Event, Israel},
  series = {SYSTOR '26}
}
```

## License

MIT
