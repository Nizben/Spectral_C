import unittest

import torch

from wan.modules.st_spectral_cpp import STSpectralCppCompressor, STSpectralCppConfig


class TestSTSpectralCpp(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def _make_compressor(
        self,
        *,
        target_budget: int,
        grid_size=(2, 2, 2),
        pool_size: int = 64,
    ) -> STSpectralCppCompressor:
        cfg = STSpectralCppConfig(
            enable=True,
            target_budget=target_budget,
            grid_size=grid_size,
            pool_size=pool_size,
            lambda_reg=0.5,
            epsilon=1e-6,
            recent_window_tokens=0,
            keep_sinks=True,
        )
        return STSpectralCppCompressor(cfg)

    def test_anchor_coverage_one_per_chunk(self) -> None:
        compressor = self._make_compressor(target_budget=16, grid_size=(2, 2, 2))

        t, h, w = 4, 4, 4
        seq_len = t * h * w
        # Strictly increasing utility gives a unique max in every chunk.
        phi = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0)
        anchor_mask = compressor._build_anchor_mask(phi, seq_len=seq_len, spatial_shape=(h, w))
        self.assertEqual(anchor_mask.shape, (1, seq_len))

        anchors_3d = anchor_mask.view(1, t, h, w)
        chunk_count = 0
        for t0 in range(0, t, 2):
            for h0 in range(0, h, 2):
                for w0 in range(0, w, 2):
                    chunk = anchors_3d[:, t0:t0 + 2, h0:h0 + 2, w0:w0 + 2]
                    self.assertEqual(int(chunk.sum().item()), 1)
                    chunk_count += 1

        self.assertEqual(int(anchor_mask.sum().item()), chunk_count)

    def test_timestep_reuse_correctness(self) -> None:
        compressor = self._make_compressor(target_budget=12, grid_size=(2, 1, 5))

        b, n_q, n_k, n_h, d_h = 1, 6, 40, 2, 4
        queries_step0 = torch.randn(b, n_q, n_h, d_h)
        keys_step0 = torch.randn(b, n_k, n_h, d_h)
        kv_cache = {}

        keep0 = compressor.compress(
            queries=queries_step0,
            keys=keys_step0,
            kv_cache=kv_cache,
            frame_seqlen=10,
            spatial_shape=(2, 5),
            sink_tokens=4,
            mandatory_recent_tokens=6,
            is_first_timestep=True,
        )
        self.assertIn("st_cached_keep_indices", kv_cache)

        # Different queries/keys should still return cached keep indices when
        # this is not the first denoising timestep.
        queries_step1 = torch.randn(b, n_q, n_h, d_h)
        keys_step1 = torch.randn(b, n_k, n_h, d_h)
        keep1 = compressor.compress(
            queries=queries_step1,
            keys=keys_step1,
            kv_cache=kv_cache,
            frame_seqlen=10,
            spatial_shape=(2, 5),
            sink_tokens=4,
            mandatory_recent_tokens=6,
            is_first_timestep=False,
        )
        self.assertTrue(torch.equal(keep0, keep1))

    def test_spectral_novelty_monotonicity(self) -> None:
        compressor = self._make_compressor(target_budget=8)

        pool = torch.randn(12, 18, dtype=torch.float32)
        basis_seed = torch.randn(6, 18, dtype=torch.float32)
        prev_residual_norm = None

        for k in range(0, basis_seed.shape[0] + 1):
            if k == 0:
                basis = torch.zeros((0, pool.shape[1]), dtype=pool.dtype)
            else:
                basis = compressor._orthonormal_rows(basis_seed[:k])

            if basis.numel() == 0:
                residual = pool
            else:
                proj = pool @ basis.transpose(0, 1)
                recon = proj @ basis
                residual = pool - recon

            residual_norm = torch.norm(residual, dim=-1)
            if prev_residual_norm is not None:
                # Adding basis vectors can only shrink (or keep) residual norms.
                self.assertTrue(torch.all(residual_norm <= prev_residual_norm + 1e-5))
            prev_residual_norm = residual_norm

    def test_cache_index_stability(self) -> None:
        compressor = self._make_compressor(target_budget=18, grid_size=(2, 2, 1))

        b, n_q, n_k, n_h, d_h = 2, 8, 48, 2, 4
        queries = torch.randn(b, n_q, n_h, d_h)
        keys = torch.randn(b, n_k, n_h, d_h)

        keep_a = compressor.compress(
            queries=queries,
            keys=keys,
            kv_cache={},
            frame_seqlen=12,
            spatial_shape=(3, 4),
            sink_tokens=6,
            mandatory_recent_tokens=8,
            is_first_timestep=True,
        )
        keep_b = compressor.compress(
            queries=queries,
            keys=keys,
            kv_cache={},
            frame_seqlen=12,
            spatial_shape=(3, 4),
            sink_tokens=6,
            mandatory_recent_tokens=8,
            is_first_timestep=True,
        )
        self.assertTrue(torch.equal(keep_a, keep_b))

        for row in keep_a:
            self.assertTrue(torch.all(row[1:] >= row[:-1]))
            self.assertEqual(torch.unique(row).numel(), row.numel())
            self.assertGreaterEqual(int(row.min().item()), 0)
            self.assertLess(int(row.max().item()), n_k)

        source_k = torch.randn(b, n_k, n_h, d_h)
        source_v = torch.randn(b, n_k, n_h, d_h)
        kv_cache = {
            "k": torch.zeros_like(source_k),
            "v": torch.zeros_like(source_v),
            "local_end_index": torch.tensor([0], dtype=torch.long),
        }

        keep_len = keep_a.shape[1]
        new_len = compressor.prune_cache_front(
            kv_cache=kv_cache,
            source_k=source_k,
            source_v=source_v,
            keep_indices=keep_a,
        )
        self.assertEqual(new_len, keep_len)
        self.assertEqual(int(kv_cache["local_end_index"].item()), keep_len)

        gather_idx = keep_a.unsqueeze(-1).unsqueeze(-1).expand(b, keep_len, n_h, d_h)
        expected_k = torch.gather(source_k, dim=1, index=gather_idx)
        expected_v = torch.gather(source_v, dim=1, index=gather_idx)
        self.assertTrue(torch.allclose(kv_cache["k"][:, :keep_len], expected_k))
        self.assertTrue(torch.allclose(kv_cache["v"][:, :keep_len], expected_v))
        self.assertTrue(torch.all(kv_cache["k"][:, keep_len:] == 0))
        self.assertTrue(torch.all(kv_cache["v"][:, keep_len:] == 0))

    def test_random_mode_keeps_sink_tail_and_fills_gap(self) -> None:
        cfg = STSpectralCppConfig(
            enable=True,
            mode="random",
            target_budget=18,
            grid_size=(4, 2, 2),
            pool_size=64,
            lambda_reg=0.5,
            epsilon=1e-6,
            recent_window_tokens=0,
            max_query_tokens=0,
            keep_sinks=True,
        )
        compressor = STSpectralCppCompressor(cfg)

        b, n_q, n_k, n_h, d_h = 1, 6, 40, 2, 4
        queries = torch.randn(b, n_q, n_h, d_h)
        keys = torch.randn(b, n_k, n_h, d_h)
        sink_tokens = 10
        mandatory_recent_tokens = 5

        keep = compressor.compress(
            queries=queries,
            keys=keys,
            kv_cache={},
            frame_seqlen=10,
            spatial_shape=(2, 5),
            sink_tokens=sink_tokens,
            mandatory_recent_tokens=mandatory_recent_tokens,
            is_first_timestep=True,
        )[0]

        self.assertEqual(keep.numel(), 18)
        self.assertEqual(torch.unique(keep).numel(), 18)
        self.assertTrue(torch.all(keep[1:] >= keep[:-1]))

        # Sink must be preserved (0..sink_tokens-1)
        sink_expected = torch.arange(0, sink_tokens, device=keep.device)
        self.assertTrue(torch.all(torch.isin(sink_expected, keep)))

        # Tail must be preserved (n_k-mandatory_recent_tokens..n_k-1)
        tail_expected = torch.arange(n_k - mandatory_recent_tokens, n_k, device=keep.device)
        self.assertTrue(torch.all(torch.isin(tail_expected, keep)))

        # The remaining picks must be from intermediate history region
        mid = keep[(keep >= sink_tokens) & (keep < (n_k - mandatory_recent_tokens))]
        self.assertEqual(mid.numel(), 18 - sink_tokens - mandatory_recent_tokens)


if __name__ == "__main__":
    unittest.main()
