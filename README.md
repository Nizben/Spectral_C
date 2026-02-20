# Deep Forcing + ST-Spectral-C++ (Comprehensive Guide)

This repository now contains a detailed implementation of a proactive KV-cache compression framework, **ST-Spectral-C++**, integrated into the Deep Forcing causal video diffusion stack.

This README is intentionally long and tutorial-style. It is designed so that:

- a newcomer to long-video diffusion can build the right mental model, and
- an experienced engineer/researcher can map every concept to concrete code quickly.

---

## Table of Contents

1. [What this project does](#1-what-this-project-does)  
2. [Core intuition: why KV cache compression exists](#2-core-intuition-why-kv-cache-compression-exists)  
3. [Deep Forcing baseline components](#3-deep-forcing-baseline-components)  
4. [ST-Spectral-C++ framework](#4-st-spectral-c-framework)  
5. [Math and tensor shapes](#5-math-and-tensor-shapes)  
6. [How ST-Spectral-C++ is implemented here](#6-how-st-spectral-c-is-implemented-here)  
7. [End-to-end call path through this codebase](#7-end-to-end-call-path-through-this-codebase)  
8. [Configuration and practical tuning](#8-configuration-and-practical-tuning)  
9. [Usage examples](#9-usage-examples)  
10. [Unit tests added for ST-Spectral-C++](#10-unit-tests-added-for-st-spectral-c)  
11. [Troubleshooting and gotchas](#11-troubleshooting-and-gotchas)  
12. [Installation and requirements](#12-installation-and-requirements)  
13. [Acknowledgements and citation](#13-acknowledgements-and-citation)

---

## 1) What this project does

Deep Forcing is a training-free long-video generation framework for autoregressive video diffusion models (Wan-based in this repo). It uses a causal attention + KV-cache inference design so generated frames can condition on prior generated frames.

The key practical problem in this family of systems is memory growth:

- each new frame adds many latent tokens,
- each transformer block stores keys and values for those tokens,
- cache eventually overflows fixed capacity.

This repository previously used Deep Sink + Participative Compression (PC).  
It now also supports ST-Spectral-C++ as a proactive overflow compressor integrated directly into causal attention.

---

## 2) Core intuition: why KV cache compression exists

In causal attention, current tokens ask questions (`Q`) and past tokens answer through cached (`K`, `V`).

If you always keep all old tokens, memory explodes.  
If you evict aggressively without structure, quality collapses.

Typical artifact patterns from poor cache policy:

- background/layout drift,
- object identity instability,
- temporal inconsistency after long horizons,
- camera-pan hallucinations when revisiting old regions.

A good compression policy should preserve:

- **coverage** (important regions across time/space),
- **relevance** (what current queries need),
- **diversity** (avoid storing redundant near-duplicates).

ST-Spectral-C++ is exactly this type of policy.

---

## 3) Deep Forcing baseline components

This codebase includes:

### Deep Sink

- Keeps a persistent sink region (early frames/tokens).
- Applies temporal RoPE adjustment so sink tokens stay aligned with current timeline.
- Improves long-term consistency.

### Participative Compression (PC)

- Computes utility from recent queries.
- Keeps top relevant old tokens and recency window.
- Reduces redundant memory while preserving useful context.

### ST-Spectral-C++ (new integration)

- Adds structured 2-stage proactive selection at overflow:
  - Stage A: spatiotemporal anchor coverage
  - Stage B: incremental spectral novelty

In `causal_model.py`, ST coexists with PC (ST branch first when enabled, PC fallback when ST disabled).  
In `causal_model_DS.py`, ST is integrated into the DS path with legacy roll fallback available if ST is disabled.

---

## 4) ST-Spectral-C++ framework

### Phase 1: Dual latency shields

1) Temporal sparsity:

- expensive selection only wakes up around overflow events.

2) Diffusion-step reuse:

- selection mask is cached and can be reused instead of recomputing expensive selection every denoising step.

### Phase 2: Fast utility scoring

Instead of computing dense attention matrices:

- collapse recent queries into an intent vector per head,
- compute candidate utility in one batched contraction.

### Phase 3: Stage A - spatiotemporal grid anchors

Treat token sequence as `(time, height, width)` lattice and chunk it.
Select one best utility token per chunk to preserve broad memory coverage.

### Phase 4: Stage B - incremental spectral novelty

Restrict candidates to high-utility pool, then greedily select tokens maximizing:

- utility + novelty

Novelty is residual norm after projection on current selected-token basis (orthonormalized).

This keeps tokens that are both useful and non-redundant.

---

## 5) Math and tensor shapes

Let:

- `queries`: `[B, N_q, H, D]`
- `keys`: `[B, N_k, H, D]`
- `B`: batch, `N_q`: recent query tokens, `N_k`: candidate keys, `H`: heads, `D`: head dim

Utility:

- `q_bar = sum(queries over N_q)` -> `[B, H, D]`
- `phi = einsum("bhd,bkhd->bk", q_bar, keys)` -> `[B, N_k]`

Anchor stage:

- reshape utility onto `(T, H, W)` token lattice
- one winner per grid chunk `(B_t, B_h, B_w)`

Spectral novelty stage:

- flatten each key token to vector dim `H * D`
- maintain orthonormal basis `U` of selected vectors
- novelty of candidate `k`:
  - `||k - Proj_U(k)||_2`
- joint:
  - `joint = phi + lambda * log(epsilon + novelty)`

---

## 6) How ST-Spectral-C++ is implemented here

## 6.1 Core module

File: `wan/modules/st_spectral_cpp.py`

### `STSpectralCppConfig`

Contains:

- `enable`
- `target_budget`
- `grid_size` (`B_t, B_h, B_w`)
- `pool_size`
- `lambda_reg`
- `epsilon`
- `recent_window_tokens`
- `keep_sinks`

### `STSpectralCppCompressor`

Main methods:

- `_update_recent_queries(kv_cache, new_queries, window_tokens)`
  - updates `kv_cache["st_recent_q"]`
- `_build_anchor_mask(phi, seq_len, spatial_shape)`
  - Stage A anchor selection mask
- `_spectral_select_single_batch(...)`
  - Stage B greedy novelty selection
- `compress(...)`
  - full selection pipeline
  - writes reuse state into `kv_cache["st_cached_keep_indices"]`
- `prune_cache_front(kv_cache, source_k, source_v, keep_indices)`
  - gathers selected rows into front of cache
  - zeroes trailing memory
  - updates `local_end_index`

## 6.2 Integration in `causal_model.py`

File: `wan/modules/causal_model.py`

Important additions:

- `CausalWanSelfAttention` now receives:
  - `PC` config and `ST` config
- In inference `kv_cache` branch:
  - detect overflow condition,
  - if ST enabled and overflow pressure exists:
    - build augmented candidate cache (`old + new`)
    - run ST compression
    - compact cache
  - else fallback to PC/rolling logic

Also:

- compatibility metadata such as `abs_frame_idx` and `topc_select_counts` remains synchronized after ST prune in this path.

## 6.3 Integration in `causal_model_DS.py`

File: `wan/modules/causal_model_DS.py`

Important additions:

- ST config and compressor added to DS causal attention class.
- On overflow:
  - build augmented candidates,
  - run ST compression,
  - compact cache.
- If ST disabled, legacy rolling behavior remains available.

## 6.4 Wrapper plumbed for usability

File: `utils/wan_wrapper.py`

`WanDiffusionWrapper` now accepts ST args:

- `st_enable`
- `st_target_budget`
- `st_grid_size`
- `st_pool_size`
- `st_lambda_reg`
- `st_epsilon`
- `st_recent_window_frames`
- `st_keep_sinks`

Compatibility aliases supported:

- uppercase forms: `ST_enable`, `ST_target_budget`, ...
- legacy forms: `Budget`, `Recent`

Default alignment behavior:

- if `st_target_budget` is missing or `<= 0`, wrapper auto-sets:
  - `st_target_budget = 1560 * budget`

This keeps ST budget aligned with existing budget convention.

---

## 7) End-to-end call path through this codebase

1) **CLI entry** -> `inference.py`

- loads OmegaConf
- injects CLI flags into `config.model_kwargs`
  - `is_ds_only`
  - `budget`
  - `recent`

2) **Pipeline init** -> `pipeline/causal_inference.py` or `pipeline/causal_diffusion_inference.py`

- constructs generator via:
  - `WanDiffusionWrapper(**model_kwargs, is_causal=True)`

3) **Wrapper dispatch** -> `utils/wan_wrapper.py`

- chooses:
  - `CausalWanModelDS` if DS-only
  - otherwise `CausalWanModel`

4) **Attention-time compression** -> causal model self-attention

- when overflow gate triggers, ST branch intercepts and compresses cache.

5) **Inference continues** with compressed cache window.

---

## 8) Configuration and practical tuning

Add ST config under `model_kwargs` in your YAML:

```yaml
model_kwargs:
  timestep_shift: 5.0
  local_attn_size: 21
  sink_size: 10

  st_enable: true
  st_target_budget: 24960
  st_grid_size: [4, 2, 2]
  st_pool_size: 1024
  st_lambda_reg: 0.5
  st_epsilon: 1e-5
  st_recent_window_frames: 4
  st_keep_sinks: true
```

### Tuning notes

- `st_target_budget`:
  - biggest quality-memory tradeoff knob.
- `st_grid_size`:
  - smaller chunks -> denser anchor coverage.
- `st_pool_size`:
  - larger -> better candidate search, higher compute.
- `st_lambda_reg`:
  - larger -> novelty favored more strongly.
- `st_recent_window_frames`:
  - larger -> broader temporal intent signal.

---

## 9) Usage examples

### Deep Sink oriented path (`is_ds_only=1`)

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --config_path configs/self_forcing_dmd/self_forcing_dmd_sink14.yaml \
  --output_folder ./output/DS \
  --checkpoint_path checkpoints/self_forcing_dmd.pt \
  --data_path ./prompts/MovieGenVideoBench_txt/line_0010.txt \
  --use_ema \
  --is_ds_only 1
```

### Deep Sink + PC oriented path

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --config_path configs/self_forcing_dmd/self_forcing_dmd_sink10.yaml \
  --output_folder ./output/DS_PC \
  --checkpoint_path checkpoints/self_forcing_dmd.pt \
  --data_path ./prompts/MovieGenVideoBench_txt/line_0043.txt \
  --use_ema
```

Helper scripts:

- `DS_inference.sh`
- `DS_PC_inference.sh`

---

## 10) Unit tests added for ST-Spectral-C++

File: `tests/test_st_spectral_cpp.py`

Covered guarantees:

1. **Anchor coverage**  
   Ensures one anchor selected per chunk in controlled setup.

2. **Timestep reuse correctness**  
   Ensures cached keep indices are reused when not first timestep.

3. **Spectral novelty monotonicity**  
   Ensures residual norms do not increase as basis expands.

4. **Cache-index stability**  
   Ensures deterministic sorted unique valid indices and correct gather-based compaction.

Typical command:

```bash
python -m unittest discover -s tests -p "test_st_spectral_cpp.py"
```

---

## 11) Troubleshooting and gotchas

### OOM still occurs

- lower `st_target_budget`
- lower `st_pool_size`
- lower `local_attn_size`

### Geometry drift appears

- increase `st_target_budget`
- reduce grid chunk size (`st_grid_size`) for stronger anchor coverage
- increase `sink_size`

### Outputs look overly constrained/repetitive

- lower `st_lambda_reg`
- increase `st_pool_size`

### Need legacy behavior quickly

- set `st_enable: false`

---

## 12) Installation and requirements

Recommended environment:

- Linux
- NVIDIA GPU (24 GB+ recommended)
- sufficient system RAM (64 GB recommended for heavy workloads)

Install:

```bash
conda create -n self_forcing python=3.10 -y
conda activate self_forcing
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

Download checkpoints:

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
```

Notes:

- Long, detailed prompts generally work better.
- `demo.py` is not the primary recommended path for Deep Forcing causal runs.

---

## 13) Acknowledgements and citation

This codebase builds on top of Self Forcing foundations:

- https://github.com/guandeh17/Self-Forcing

If this repository is useful for your research, cite:

```bibtex
@article{yi2025deep,
  title={Deep Forcing: Training-Free Long Video Generation with Deep Sink and Participative Compression},
  author={Yi, Jung and Jang, Wooseok and Cho, Paul Hyunbin and Nam, Jisu and Yoon, Heeji and Kim, Seungryong},
  journal={arXiv preprint arXiv:2512.05081},
  year={2025}
}
```

Project links:

- Paper: https://arxiv.org/abs/2512.05081
- Website: https://cvlab-kaist.github.io/DeepForcing/

---

## 14) Precise phase-to-code alignment table

| ST-Spectral-C++ phase | Primary code location | Key symbols/methods | Runtime state touched |
|---|---|---|---|
| Phase 1: temporal sparsity trigger | `wan/modules/causal_model.py`, `wan/modules/causal_model_DS.py` | overflow/roll condition branch in `CausalWanSelfAttention.forward` | `kv_cache["global_end_index"]`, `kv_cache["local_end_index"]` |
| Phase 1: reuse state | `wan/modules/st_spectral_cpp.py` | `compress(...)` early-return on cached indices | `kv_cache["st_cached_keep_indices"]` |
| Phase 2: utility scoring | `wan/modules/st_spectral_cpp.py` | `q_bar = queries.sum(...)`, `einsum("bhd,bkhd->bk", ...)` | temporary `phi` |
| Phase 3: anchor coverage | `wan/modules/st_spectral_cpp.py` | `_build_anchor_mask(...)`, `_map_chunk_local_to_global(...)` | temporary anchor mask |
| Phase 4: novelty selection | `wan/modules/st_spectral_cpp.py` | `_spectral_select_single_batch(...)`, `_orthonormal_rows(...)` | temporary basis/projections |
| Final cache rewrite | `wan/modules/st_spectral_cpp.py` | `prune_cache_front(...)` | `kv_cache["k"]`, `kv_cache["v"]`, `kv_cache["local_end_index"]` |
| Wrapper plumbing | `utils/wan_wrapper.py` | `WanDiffusionWrapper.__init__` ST kwargs and aliases | wrapper-level constructor args |
| Pipeline flow | `pipeline/causal_inference.py`, `pipeline/causal_diffusion_inference.py` | `WanDiffusionWrapper(**model_kwargs, is_causal=True)` | `model_kwargs` from config/CLI |

---

## 15) Exact overflow gate and data flow explained

In attention inference mode (`kv_cache is not None`), the model computes:

- `num_new_tokens`: number of incoming tokens for current block
- `prev_local_end`: currently filled cache tokens
- `kv_cache_size`: allocated capacity
- `prev_global_end`: previous global progression index
- `current_end`: current block end index

Overflow is considered when local-attention path is active and:

- `current_end > prev_global_end` (progressing causal frontier), and
- `num_new_tokens + prev_local_end > kv_cache_size` (capacity exceeded)

When ST is enabled:

1. Build augmented candidates `K_aug` and `V_aug` from existing + new tokens.
2. Update recent query buffer (`st_recent_q`).
3. Run `compress(...)` to obtain keep indices.
4. Run `prune_cache_front(...)` to compact cache.
5. Continue normal attention over resulting window (`key_win`, `val_win`).

When ST is disabled:

- fallback path uses existing PC logic (main model) or rolling logic.

---

## 16) Internal cache keys you should know

These keys appear in causal inference KV-cache dictionaries:

- `k`: cached keys, shape `[B, cap, heads, head_dim]`
- `v`: cached values, same shape pattern as `k`
- `global_end_index`: progression marker in flattened sequence
- `local_end_index`: filled length in cache

ST-specific:

- `st_recent_q`: rolling recent query buffer used for utility intent
- `st_cached_keep_indices`: cached keep indices for reuse logic

Main-model compatibility bookkeeping:

- `abs_frame_idx`
- `topc_select_counts`

Cross-attention cache (separate structure) includes:

- `k`, `v`, and `is_init`

---

## 17) Complexity intuition (rough)

Let:

- `K = number of candidate tokens`
- `P = pool_size`
- `M = target_budget`
- `d = flattened token feature dimension (heads * head_dim)`

Dominant pieces:

- Utility scoring: roughly `O(K * d)`
- Anchor scan: linear in token count/chunks
- Spectral stage (greedy):
  - projection/reconstruction over pool and growing basis
  - roughly `O(M * P * d)` scale behavior

Because ST is gated by overflow events and pool-restricted, wall-clock overhead is usually much lower than running full heavy selection every denoising step.

---

## 18) Advanced tuning recipes

### Profile-driven memory reduction

Goal: fit bigger outputs under tight VRAM.

- lower `st_target_budget` first
- then lower `st_pool_size`
- if needed reduce `local_attn_size`
- keep `sink_size` non-zero to avoid long-horizon collapse

### Quality-first long-horizon stability

Goal: maximize temporal consistency.

- increase `st_target_budget`
- keep `st_keep_sinks: true`
- use finer `st_grid_size` chunks for stronger Stage A coverage
- keep moderate novelty (`st_lambda_reg` around baseline and tune gradually)

### Motion diversity recovery

Goal: avoid over-constrained memory.

- slightly lower `st_lambda_reg` if novelty term over-dominates selection dynamics
- increase `st_pool_size` to expose richer candidate set
- verify budget not too small relative to scene complexity

---

## 19) Step-by-step onboarding path (recommended for new contributors)

1. Read this README sections in order through Section 8.
2. Open `wan/modules/st_spectral_cpp.py` and inspect:
   - config class,
   - `compress(...)`,
   - `prune_cache_front(...)`.
3. Open `wan/modules/causal_model.py` and `wan/modules/causal_model_DS.py`:
   - locate overflow branch in `CausalWanSelfAttention.forward`.
4. Open `utils/wan_wrapper.py`:
   - verify constructor argument plumbing and aliases.
5. Run/inspect tests in `tests/test_st_spectral_cpp.py`.
6. Modify one ST hyperparameter in YAML and re-run inference.

This sequence helps you understand concept -> implementation -> behavior with minimal confusion.

---

## 20) FAQ

### Q1) Is ST-Spectral-C++ training-time or inference-time?

Inference-time KV-cache path in causal attention.

### Q2) Do I need to retrain checkpoints?

No. This is a runtime cache-management mechanism.

### Q3) Does ST replace Deep Sink?

No. Sink preservation remains important. ST works with sink-aware constraints.

### Q4) Does ST replace PC?

Not globally. In the main causal model path, ST and PC coexist with ST-enabled branch and PC fallback behavior.

### Q5) What if I want baseline behavior for ablation?

Set `st_enable: false`.

### Q6) Which file should I read first for implementation details?

`wan/modules/st_spectral_cpp.py`

---

## 21) Practical warning notes

- Keep parameter naming consistent: use lowercase `st_*` in YAML for clarity (aliases are supported, but consistency helps debugging).
- If you disable ST, confirm expected fallback path (PC vs rolling) based on which model (`CausalWanModel` vs `CausalWanModelDS`) you are using.
- Ensure `local_attn_size` is set intentionally, since cache capacity is derived from it in causal pipelines.
- If results change unexpectedly, inspect whether your effective budget was auto-derived from `budget` (wrapper behavior) or explicitly set through `st_target_budget`.

