import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class STSpectralCppConfig:
    def __init__(
        self,
        enable: bool = False,
        target_budget: int = 0,
        grid_size: Tuple[int, int, int] = (4, 2, 2),
        pool_size: int = 1024,
        lambda_reg: float = 0.5,
        epsilon: float = 1e-5,
        recent_window_tokens: int = 0,
        max_query_tokens: int = 2048,
        keep_sinks: bool = True,
    ):
        self.enable = bool(enable)
        self.target_budget = int(target_budget)
        self.grid_size = tuple(int(x) for x in grid_size)
        self.pool_size = int(pool_size)
        self.lambda_reg = float(lambda_reg)
        self.epsilon = float(epsilon)
        self.recent_window_tokens = int(recent_window_tokens)
        self.max_query_tokens = int(max_query_tokens)
        self.keep_sinks = bool(keep_sinks)


class STSpectralCppCompressor:
    def __init__(self, cfg: STSpectralCppConfig):
        self.cfg = cfg

    @staticmethod
    def _orthonormal_rows(vectors: torch.Tensor) -> torch.Tensor:
        if vectors.numel() == 0:
            return vectors
        # Build row-orthonormal basis with a reduced QR on transposed matrix.
        q, _ = torch.linalg.qr(vectors.transpose(0, 1), mode="reduced")
        return q.transpose(0, 1)

    @staticmethod
    def _map_chunk_local_to_global(
        local_idx: torch.Tensor,
        t_start: int,
        x_start: int,
        y_start: int,
        chunk_t: int,
        chunk_h: int,
        chunk_w: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        area = chunk_h * chunk_w
        t_off = torch.div(local_idx, area, rounding_mode="floor")
        rem = local_idx % area
        x_off = torch.div(rem, chunk_w, rounding_mode="floor")
        y_off = rem % chunk_w

        t = t_start + t_off
        x = x_start + x_off
        y = y_start + y_off
        return (t * height + x) * width + y

    @staticmethod
    def _update_recent_queries(kv_cache: dict, new_queries: torch.Tensor, window_tokens: int) -> torch.Tensor:
        new_queries = new_queries.detach()
        if window_tokens <= 0:
            kv_cache["st_recent_q"] = new_queries
            return new_queries

        prev = kv_cache.get("st_recent_q", None)
        if prev is None:
            merged = new_queries
        else:
            merged = torch.cat([prev, new_queries], dim=1)
        if merged.shape[1] > window_tokens:
            merged = merged[:, -window_tokens:]
        kv_cache["st_recent_q"] = merged
        return merged

    @staticmethod
    def _topk_indices(values: torch.Tensor, k: int) -> torch.Tensor:
        if values.numel() == 0 or k <= 0:
            return torch.zeros(0, dtype=torch.long, device=values.device)
        k = min(k, values.numel())
        return torch.topk(values, k=k, dim=0).indices

    def _build_anchor_mask(
        self,
        phi: torch.Tensor,
        seq_len: int,
        spatial_shape: Tuple[int, int],
    ) -> torch.Tensor:
        bsz = phi.shape[0]
        height, width = spatial_shape
        frame_tokens = height * width
        time_frames = math.ceil(seq_len / frame_tokens)
        
        # Pad phi to form a perfect 3D volume
        padded = torch.full((bsz, time_frames * frame_tokens), -float("inf"), device=phi.device, dtype=phi.dtype)
        padded[:, :seq_len] = phi
        
        # Reshape for max_pool3d: [B, Channels(1), T, H, W]
        phi_3d = padded.view(bsz, 1, time_frames, height, width).to(torch.float32)
        
        bt, bh, bw = [max(1, x) for x in self.cfg.grid_size]
        
        # F.max_pool3d natively returns the flattened 1D indices! O(1) Python overhead.
        _, indices = F.max_pool3d(phi_3d, kernel_size=(bt, bh, bw), stride=(bt, bh, bw), return_indices=True)
        
        # Flatten and filter invalid indices
        valid_indices = indices.view(bsz, -1)
        valid_indices = valid_indices[valid_indices < seq_len]
        
        anchor_mask = torch.zeros((bsz, seq_len), dtype=torch.bool, device=phi.device)
        for b in range(bsz):
            b_indices = indices[b].flatten()
            b_valid = b_indices[b_indices < seq_len]
            anchor_mask[b, b_valid] = True
            
        return anchor_mask

    def _spectral_select_single_batch(
        self,
        phi_b: torch.Tensor,
        keys_b: torch.Tensor,
        selected_idx: torch.Tensor,
        budget: int,
    ) -> torch.Tensor:
        n_k, n_h, d_h = keys_b.shape
        flat_dim = n_h * d_h

        if selected_idx.numel() >= budget:
            return torch.sort(selected_idx[:budget]).values

        selected_mask = torch.zeros(n_k, dtype=torch.bool, device=keys_b.device)
        selected_mask[selected_idx] = True
        remaining = budget - int(selected_idx.numel())

        candidate_idx = torch.where(~selected_mask)[0]
        if candidate_idx.numel() == 0:
            return torch.sort(selected_idx).values

        k_pool = min(self.cfg.pool_size, candidate_idx.numel())
        pool_pick_local = self._topk_indices(phi_b[candidate_idx], k_pool)
        pool_idx = candidate_idx[pool_pick_local]

        # OPTIMIZATION 1: Truncate the QR basis to prevent GPU stalling
        if selected_idx.numel() > 0:
            qr_seed_idx = selected_idx
            if qr_seed_idx.numel() > 256: 
                # Take top 256 anchors by utility to seed the basis
                top_seed_local = self._topk_indices(phi_b[qr_seed_idx], 256)
                qr_seed_idx = qr_seed_idx[top_seed_local]
            basis_rows = keys_b[qr_seed_idx].reshape(-1, flat_dim).to(torch.float32)
            basis = self._orthonormal_rows(basis_rows)
        else:
            basis = torch.zeros((0, flat_dim), dtype=torch.float32, device=keys_b.device)

        chosen = [selected_idx]
        
        # OPTIMIZATION 2: Limit the iterative Gram-Schmidt to max 128 iterations.
        # This captures the most crucial novelty without looping thousands of times in Python.
        spectral_iterations = min(remaining, 128) 
        
        while spectral_iterations > 0 and pool_idx.numel() > 0:
            pool_vecs = keys_b[pool_idx].reshape(-1, flat_dim).to(torch.float32)
            if basis.numel() == 0:
                residuals = pool_vecs
            else:
                projections = pool_vecs @ basis.transpose(0, 1)
                recon = projections @ basis
                residuals = pool_vecs - recon

            novelty = torch.norm(residuals, dim=-1)
            utility = phi_b[pool_idx]
            joint = utility + self.cfg.lambda_reg * torch.log(self.cfg.epsilon + novelty)
            best_local = int(torch.argmax(joint).item())
            best_global = pool_idx[best_local:best_local + 1]
            chosen.append(best_global)

            vec = residuals[best_local:best_local + 1]
            if torch.isfinite(vec).all() and vec.norm() > self.cfg.epsilon:
                vec = F.normalize(vec, p=2, dim=-1, eps=self.cfg.epsilon)
                basis = torch.cat([basis, vec], dim=0)

            pool_idx = torch.cat([pool_idx[:best_local], pool_idx[best_local + 1:]], dim=0)
            remaining -= 1
            spectral_iterations -= 1

        selected = torch.cat(chosen, dim=0) if len(chosen) > 0 else torch.zeros(0, dtype=torch.long, device=keys_b.device)
        selected = torch.unique(selected, sorted=True)
        
        # OPTIMIZATION 3: If budget remains, fill instantly with remaining Top-Utility (\phi)
        if selected.numel() < budget:
            selected_mask = torch.zeros(n_k, dtype=torch.bool, device=keys_b.device)
            selected_mask[selected] = True
            tail_candidates = torch.where(~selected_mask)[0]
            fill = min(budget - selected.numel(), tail_candidates.numel())
            if fill > 0:
                fill_local = self._topk_indices(phi_b[tail_candidates], fill)
                selected = torch.cat([selected, tail_candidates[fill_local]], dim=0)
                selected = torch.unique(selected, sorted=True)

        return torch.sort(selected[:budget]).values

    def compress(
        self,
        *,
        queries: torch.Tensor,
        keys: torch.Tensor,
        kv_cache: dict,
        frame_seqlen: int,
        spatial_shape: Tuple[int, int],
        sink_tokens: int,
        mandatory_recent_tokens: int,
        is_first_timestep: bool,
    ) -> torch.Tensor:
        # queries: [B, N_q, H, D], keys: [B, N_k, H, D]
        bsz, n_k = keys.shape[:2]
        device = keys.device

        if (not is_first_timestep) and ("st_cached_keep_indices" in kv_cache):
            cached = kv_cache["st_cached_keep_indices"]
            if isinstance(cached, torch.Tensor) and cached.shape[0] == bsz:
                return cached

        # ==========================================================
        # Fast-path: if budget is fully consumed by sinks + mandatory recent,
        # there is no anchor/novelty selection to do. Avoid O(N_q * N_k) scoring.
        # ==========================================================
        target_budget = self.cfg.target_budget if self.cfg.target_budget > 0 else n_k
        target_budget = min(int(target_budget), n_k)
        target_budget = max(1, target_budget)

        mandatory_recent_tokens = int(max(0, mandatory_recent_tokens))
        mandatory_recent_tokens = min(mandatory_recent_tokens, target_budget, n_k)

        keep_sink_tokens = int(max(0, sink_tokens if self.cfg.keep_sinks else 0))
        keep_sink_tokens = min(keep_sink_tokens, n_k - mandatory_recent_tokens)
        keep_sink_tokens = min(keep_sink_tokens, target_budget - mandatory_recent_tokens)

        if target_budget == n_k:
            keep = torch.arange(n_k, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
            kv_cache["st_cached_keep_indices"] = keep
            return keep

        if keep_sink_tokens + mandatory_recent_tokens == target_budget:
            sink_idx = torch.arange(keep_sink_tokens, device=device, dtype=torch.long)
            recent_idx = torch.arange(n_k - mandatory_recent_tokens, n_k, device=device, dtype=torch.long)
            keep_1d = torch.cat([sink_idx, recent_idx], dim=0)
            keep = keep_1d.unsqueeze(0).expand(bsz, -1)
            kv_cache["st_cached_keep_indices"] = keep
            return keep

        # ==========================================================
        # FIXED PHASE 2: Chunked Max-Fusion Utility Scoring
        # Preserves high-entropy spatial details (like the driver)
        # ==========================================================
        max_q = int(getattr(self.cfg, "max_query_tokens", 0) or 0)
        if max_q > 0 and queries.shape[1] > max_q:
            # Keep the most recent queries only. This dramatically reduces the
            # O(N_q * N_k) cost of utility scoring without changing the cache layout.
            queries = queries[:, -max_q:]

        B_q, N_q, H_q, D_q = queries.shape
        
        # Initialize phi with negative infinity
        phi = torch.full((B_q, n_k), -float('inf'), device=device, dtype=torch.float32)
        
        # Transpose for batch-matrix multiplication: [B, H, N, D]
        q_flat = queries.to(torch.float32).transpose(1, 2)
        k_flat = keys.to(torch.float32).transpose(1, 2)
        
        # Process in chunks to maintain Latency-Lock and avoid OOM
        chunk_size = 512
        for start in range(0, N_q, chunk_size):
            end = min(start + chunk_size, N_q)
            q_chunk = q_flat[:, :, start:end, :] 
            
            # Dot product across D, sum across H: yields [B, Chunk, N_k]
            scores = torch.einsum('bhqd,bhkd->bqk', q_chunk, k_flat)
            
            # Max across the temporal/spatial queries in this chunk
            chunk_max, _ = scores.max(dim=1)
            phi = torch.maximum(phi, chunk_max)

        selected_mask = torch.zeros((bsz, n_k), dtype=torch.bool, device=device)
        if keep_sink_tokens > 0:
            selected_mask[:, :keep_sink_tokens] = True
        if mandatory_recent_tokens > 0:
            selected_mask[:, n_k - mandatory_recent_tokens:] = True

        anchor_mask = self._build_anchor_mask(phi, n_k, spatial_shape=spatial_shape)

        # Add anchors by utility order until budget is reached.
        for b in range(bsz):
            current_count = int(selected_mask[b].sum().item())
            room = target_budget - current_count
            if room <= 0:
                continue
            anchor_candidates = torch.where(anchor_mask[b] & (~selected_mask[b]))[0]
            if anchor_candidates.numel() == 0:
                continue
            pick = min(room, anchor_candidates.numel())
            local = self._topk_indices(phi[b, anchor_candidates], pick)
            chosen = anchor_candidates[local]
            selected_mask[b, chosen] = True

        # Stage B spectral fill for each batch item.
        keep = torch.zeros((bsz, target_budget), dtype=torch.long, device=device)
        for b in range(bsz):
            initial = torch.where(selected_mask[b])[0]
            chosen = self._spectral_select_single_batch(
                phi_b=phi[b],
                keys_b=keys[b],
                selected_idx=initial,
                budget=target_budget,
            )
            keep[b, :chosen.numel()] = chosen
            if chosen.numel() < target_budget:
                keep[b, chosen.numel():] = chosen[-1] if chosen.numel() > 0 else 0

        keep = torch.sort(keep, dim=-1).values
        kv_cache["st_cached_keep_indices"] = keep
        return keep

    @staticmethod
    def prune_cache_front(
        kv_cache: dict,
        source_k: torch.Tensor,
        source_v: torch.Tensor,
        keep_indices: torch.Tensor,
    ) -> int:
        # source_k/source_v: [B, T_src, H, D], keep_indices: [B, K]
        bsz, keep_len = keep_indices.shape
        _, _, n_h, d_h = source_k.shape
        device = source_k.device

        gather_idx = keep_indices.unsqueeze(-1).unsqueeze(-1).expand(bsz, keep_len, n_h, d_h)
        new_k = torch.gather(source_k, dim=1, index=gather_idx)
        new_v = torch.gather(source_v, dim=1, index=gather_idx)

        kv_cache["k"][:, :keep_len] = new_k
        kv_cache["v"][:, :keep_len] = new_v
        if kv_cache["k"].shape[1] > keep_len:
            kv_cache["k"][:, keep_len:] = 0
            kv_cache["v"][:, keep_len:] = 0

        if "local_end_index" in kv_cache and torch.is_tensor(kv_cache["local_end_index"]):
            kv_cache["local_end_index"].fill_(keep_len)
        else:
            kv_cache["local_end_index"] = torch.tensor([keep_len], dtype=torch.long, device=device)
        return keep_len
