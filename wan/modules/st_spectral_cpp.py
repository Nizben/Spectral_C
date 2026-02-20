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
        keep_sinks: bool = True,
    ):
        self.enable = bool(enable)
        self.target_budget = int(target_budget)
        self.grid_size = tuple(int(x) for x in grid_size)
        self.pool_size = int(pool_size)
        self.lambda_reg = float(lambda_reg)
        self.epsilon = float(epsilon)
        self.recent_window_tokens = int(recent_window_tokens)
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
        if frame_tokens <= 0:
            return torch.zeros((bsz, seq_len), dtype=torch.bool, device=phi.device)

        time_frames = math.ceil(seq_len / frame_tokens)
        padded = torch.full(
            (bsz, time_frames * frame_tokens),
            -float("inf"),
            device=phi.device,
            dtype=phi.dtype,
        )
        padded[:, :seq_len] = phi
        phi_3d = padded.view(bsz, time_frames, height, width)

        bt, bh, bw = self.cfg.grid_size
        bt = max(1, bt)
        bh = max(1, bh)
        bw = max(1, bw)

        anchor_mask = torch.zeros((bsz, seq_len), dtype=torch.bool, device=phi.device)
        for t0 in range(0, time_frames, bt):
            t1 = min(time_frames, t0 + bt)
            for h0 in range(0, height, bh):
                h1 = min(height, h0 + bh)
                for w0 in range(0, width, bw):
                    w1 = min(width, w0 + bw)
                    chunk = phi_3d[:, t0:t1, h0:h1, w0:w1].reshape(bsz, -1)
                    local = chunk.argmax(dim=-1)
                    global_idx = self._map_chunk_local_to_global(
                        local,
                        t_start=t0,
                        x_start=h0,
                        y_start=w0,
                        chunk_t=(t1 - t0),
                        chunk_h=(h1 - h0),
                        chunk_w=(w1 - w0),
                        height=height,
                        width=width,
                    )
                    valid = global_idx < seq_len
                    if valid.any():
                        rows = torch.arange(bsz, device=phi.device)[valid]
                        cols = global_idx[valid]
                        anchor_mask[rows, cols] = True
        return anchor_mask

    def _spectral_select_single_batch(
        self,
        phi_b: torch.Tensor,
        keys_b: torch.Tensor,
        selected_idx: torch.Tensor,
        budget: int,
    ) -> torch.Tensor:
        # keys_b: [N_k, H, D]
        n_k, n_h, d_h = keys_b.shape
        flat_dim = n_h * d_h

        if selected_idx.numel() >= budget:
            selected_idx = selected_idx[:budget]
            return torch.sort(selected_idx).values

        selected_mask = torch.zeros(n_k, dtype=torch.bool, device=keys_b.device)
        selected_mask[selected_idx] = True
        remaining = budget - int(selected_idx.numel())
        if remaining <= 0:
            return torch.sort(selected_idx).values

        candidate_idx = torch.where(~selected_mask)[0]
        if candidate_idx.numel() == 0:
            return torch.sort(selected_idx).values

        k_pool = min(self.cfg.pool_size, candidate_idx.numel())
        pool_pick_local = self._topk_indices(phi_b[candidate_idx], k_pool)
        pool_idx = candidate_idx[pool_pick_local]

        if selected_idx.numel() > 0:
            basis_rows = keys_b[selected_idx].reshape(-1, flat_dim).to(torch.float32)
            basis = self._orthonormal_rows(basis_rows)
        else:
            basis = torch.zeros((0, flat_dim), dtype=torch.float32, device=keys_b.device)

        chosen = [selected_idx]
        while remaining > 0 and pool_idx.numel() > 0:
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
            if torch.isfinite(vec).all():
                vec = F.normalize(vec, p=2, dim=-1, eps=self.cfg.epsilon)
                basis = torch.cat([basis, vec], dim=0)

            pool_idx = torch.cat([pool_idx[:best_local], pool_idx[best_local + 1:]], dim=0)
            remaining -= 1

        selected = torch.cat(chosen, dim=0) if len(chosen) > 0 else torch.zeros(0, dtype=torch.long, device=keys_b.device)
        selected = torch.unique(selected, sorted=True)
        if selected.numel() < budget:
            selected_mask = torch.zeros(n_k, dtype=torch.bool, device=keys_b.device)
            selected_mask[selected] = True
            tail_candidates = torch.where(~selected_mask)[0]
            fill = min(budget - selected.numel(), tail_candidates.numel())
            if fill > 0:
                fill_local = self._topk_indices(phi_b[tail_candidates], fill)
                selected = torch.cat([selected, tail_candidates[fill_local]], dim=0)
                selected = torch.unique(selected, sorted=True)

        if selected.numel() > budget:
            selected = selected[:budget]
        return torch.sort(selected).values

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

        q_bar = queries.to(torch.float32).sum(dim=1)  # [B, H, D]
        phi = torch.einsum("bhd,bkhd->bk", q_bar, keys.to(torch.float32))

        target_budget = self.cfg.target_budget if self.cfg.target_budget > 0 else n_k
        target_budget = min(int(target_budget), n_k)
        target_budget = max(1, target_budget)

        # Keep most recent tokens to preserve current block fidelity.
        mandatory_recent_tokens = int(max(0, mandatory_recent_tokens))
        mandatory_recent_tokens = min(mandatory_recent_tokens, target_budget, n_k)

        keep_sink_tokens = int(max(0, sink_tokens if self.cfg.keep_sinks else 0))
        keep_sink_tokens = min(keep_sink_tokens, n_k - mandatory_recent_tokens)
        keep_sink_tokens = min(keep_sink_tokens, target_budget - mandatory_recent_tokens)

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
