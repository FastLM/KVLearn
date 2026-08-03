# KVLearn

Learning-based KV cache **retention** for disaggregated LLM serving.

This repository implements the control-plane from the paper
*To Keep or Not to Keep: Learning KV Cache Retention in Disaggregated LLM Serving Systems*:

| Component | Role |
|-----------|------|
| **PRP** | Prefix Reuse Predictor — `P̂(b) = f_θ(x(b)) ∈ [0,1]`, updated online from delayed reuse labels |
| **CARS** | Cost-Aware Retention Score — `CARS(b) = P̂(b)·(R(b)−T(b)) − U(b,Δt)` |
| **ATC** | Adaptive Threshold Controller — adapts `θ` from pool pressure and hit rate |

Admission and eviction run in the KV-pool coordinator path.

## Build

**Make (no CMake required):**

```bash
make -j
make test
./build/kvlearn_sim [n_requests] [zipf_alpha] [pool_GiB]
```

**CMake (optional):**

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
@inproceedings{liu2026kvlearn,
  author    = {Dong Liu and Yanxuan Yu and Eric Jiang and Shu Wang and Ying Nian Wu},
  title     = {To Keep or Not to Keep: Learning KV Cache Retention in Disaggregated LLM Serving Systems},
  booktitle = {Proceedings of the 19th ACM International Systems and Storage Conference (SYSTOR '26)},
  series     = {SYSTOR '26},
  year       = {2026},
  publisher  = {Association for Computing Machinery},
  address    = {New York, NY, USA}
}
```

## License

MIT
