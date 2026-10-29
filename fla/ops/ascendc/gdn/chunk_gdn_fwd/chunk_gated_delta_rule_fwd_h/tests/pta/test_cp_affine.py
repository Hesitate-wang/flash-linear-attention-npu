"""CPU reference tests for the chunk-parallel (CP) affine prefix path of
chunk_gated_delta_rule_fwd_h.

The CP kernel splits the chunk axis across ranks and runs three phases:

  phase 1: each rank builds the affine summary (B_bar, A_bar) of its own chunk
           range, starting from (I, 0);
  phase 2: each rank replays the summaries of all prior ranks to obtain the
           real start state S_in = B_trans @ H0 + S_local;
  phase 3: each rank replays its own chunks with S_in and writes h / v_new /
           final_state.

This file validates the math offline (no NPU, no torch):
  * phase 1 uses the left-projection rewrite  P = W @ M; M' = D*M - K^T(q*P)
    instead of materializing B_c = D*I - (q*K)^T @ W, and both must agree;
  * rank summaries / prefixes / replay must match the serial recurrence;
  * the rank partition (R = min(core_num, num_chunks)) never yields empty ranks;
  * an FP32 error model estimates whether the 1e-4 / 99.99% PTA threshold is
    reachable before the device run.

Run: python3 test_cp_affine.py
"""

import unittest

import numpy as np


def _gate_vector(rng, length, scale=0.01):
    """Chunk-local log decay: monotonically decreasing, last value = g_last."""
    return -np.cumsum(rng.uniform(0.0, scale, size=length))


def make_case(rng, num_chunks, chunk_size, seqlen, k_dim=16, v_dim=8):
    """Build per-chunk inputs; the tail chunk is shorter than chunk_size."""
    chunks = []
    for chunk in range(num_chunks):
        begin = chunk * chunk_size
        end = min(begin + chunk_size, seqlen)
        length = end - begin
        k_c = rng.standard_normal((length, k_dim)) * 0.05
        w_c = rng.standard_normal((length, k_dim)) * 0.05
        u_c = rng.standard_normal((length, v_dim)) * 0.05
        g_c = _gate_vector(rng, length)
        chunks.append({"k": k_c, "w": w_c, "u": u_c, "g": g_c})
    return chunks


def chunk_affine(k_c, w_c, u_c, g_c):
    """Explicit affine (B_c, A_c) of one chunk, FP64."""
    q = np.exp(g_c[-1] - g_c)
    decay = np.exp(g_c[-1])
    qk = q[:, None] * k_c
    b = decay * np.eye(k_c.shape[1]) - qk.T @ w_c
    a = qk.T @ u_c
    return b, a, q, decay


def run_chunk(state, chunk):
    """Serial recurrence: v_new = u - w @ h; h' = D*h + K^T(q*v_new)."""
    q = np.exp(chunk["g"][-1] - chunk["g"])
    decay = np.exp(chunk["g"][-1])
    v_new = chunk["u"] - chunk["w"] @ state
    next_state = decay * state + chunk["k"].T @ (v_new * q[:, None])
    return next_state, v_new


def serial_run(chunks, h0):
    starts, v_new_list = [], []
    state = h0.copy()
    for chunk in chunks:
        starts.append(state.copy())
        state, v_new = run_chunk(state, chunk)
        v_new_list.append(v_new)
    return starts, v_new_list, state


def partition(num_chunks, ranks):
    """Contiguous chunk ranges, same formula as the kernel."""
    base, rem = divmod(num_chunks, ranks)
    ranges, begin = [], 0
    for rank in range(ranks):
        count = base + (1 if rank < rem else 0)
        ranges.append((begin, begin + count))
        begin += count
    return ranges


def rank_count(core_num, num_chunks):
    """R = min(core_num, num_chunks): every rank owns at least one chunk."""
    return min(core_num, num_chunks)


def local_summary_explicit(chunks):
    """Phase 1 with the pseudo-code form M = B_c @ M (FP64)."""
    k_dim = chunks[0]["k"].shape[1]
    v_dim = chunks[0]["u"].shape[1]
    m = np.eye(k_dim)
    s = np.zeros((k_dim, v_dim))
    for chunk in chunks:
        b, a, _, _ = chunk_affine(chunk["k"], chunk["w"], chunk["u"], chunk["g"])
        m = b @ m
        s = b @ s + a
    return m, s


def local_summary_projected(chunks):
    """Phase 1 with the left-projection rewrite (FP64).

    P = W @ M;  M' = D*M - K^T(q*P)          (== (D*I - (q*K)^T W) @ M)
    S' = D*S + K^T(q*(U - W@S))             (same recurrence as the serial path)
    """
    k_dim = chunks[0]["k"].shape[1]
    v_dim = chunks[0]["u"].shape[1]
    state = np.concatenate([np.eye(k_dim), np.zeros((k_dim, v_dim))], axis=1)
    for chunk in chunks:
        q = np.exp(chunk["g"][-1] - chunk["g"])
        decay = np.exp(chunk["g"][-1])
        proj = chunk["w"] @ state                      # [T, K+V]
        v_new = chunk["u"] - proj[:, k_dim:]
        rq = -q[:, None] * proj[:, :k_dim]             # Rp = -(q * P)
        rv = q[:, None] * v_new                        # Rv = q * v_new
        # M' = D*M - (q*K)^T P = D*M + K^T(Rp); S' = D*S + K^T(Rv).
        delta = chunk["k"].T @ np.concatenate([rq, rv], axis=1)
        state = decay * state + delta
    return state[:, :k_dim].copy(), state[:, k_dim:].copy()


def prefix_state(summaries, rank, h0):
    """Phase 2: left-multiply prior rank summaries in ascending order."""
    k_dim, v_dim = h0.shape
    b_trans = np.eye(k_dim)
    s_local = np.zeros((k_dim, v_dim))
    for prior in range(rank):
        b_bar, a_bar = summaries[prior]
        b_trans = b_bar @ b_trans
        s_local = b_bar @ s_local + a_bar
    return b_trans @ h0 + s_local


def cp_run(chunks, h0, ranks):
    """Full three-phase CP pipeline, FP64."""
    slices = partition(len(chunks), ranks)
    summaries = [local_summary_projected(chunks[begin:end]) for begin, end in slices]
    starts, v_new_list = [None] * len(chunks), [None] * len(chunks)
    final_state = None
    for rank, (begin, end) in enumerate(slices):
        state = prefix_state(summaries, rank, h0)
        for idx in range(begin, end):
            starts[idx] = state.copy()
            state, v_new = run_chunk(state, chunks[idx])
            v_new_list[idx] = v_new
        if rank == ranks - 1:
            final_state = state
    return starts, v_new_list, final_state


def assert_allclose(test, expected, actual, tol=1e-12):
    test.assertEqual(expected.shape, actual.shape)
    diff = np.max(np.abs(expected - actual))
    scale = max(1.0, float(np.max(np.abs(expected))))
    test.assertLessEqual(diff / scale, tol, "max abs diff %g (scaled)" % diff)


class CpAffineTest(unittest.TestCase):
    def test_rank_partition_has_no_empty_rank(self):
        for num_chunks, core_num in ((2, 2), (5, 3), (5, 24), (24, 24), (1024, 24)):
            ranks = rank_count(core_num, num_chunks)
            self.assertLessEqual(ranks, core_num)
            slices = partition(num_chunks, ranks)
            self.assertEqual(slices[0][0], 0)
            self.assertEqual(slices[-1][1], num_chunks)
            for begin, end in slices:
                self.assertGreater(end - begin, 0)
            for prev, nxt in zip(slices, slices[1:]):
                self.assertEqual(prev[1], nxt[0])

    def test_left_projection_matches_explicit_affine(self):
        rng = np.random.default_rng(7)
        for chunk_size in (32, 64, 128):
            chunks = make_case(rng, num_chunks=4, chunk_size=chunk_size,
                               seqlen=chunk_size * 4 - 5)
            m_exp, s_exp = local_summary_explicit(chunks[:2])
            m_prj, s_prj = local_summary_projected(chunks[:2])
            assert_allclose(self, m_exp, m_prj)
            assert_allclose(self, s_exp, s_prj)

    def test_rank_summaries_match_serial(self):
        rng = np.random.default_rng(11)
        for chunk_size in (32, 64, 128):
            chunks = make_case(rng, num_chunks=5, chunk_size=chunk_size,
                               seqlen=chunk_size * 5 - 13, k_dim=16, v_dim=8)
            h0 = rng.standard_normal((chunks[0]["k"].shape[1],
                                      chunks[0]["u"].shape[1])) * 0.1
            ref_starts, ref_vnew, ref_final = serial_run(chunks, h0)
            for ranks in (1, 2, 3, 5):
                got_starts, got_vnew, got_final = cp_run(chunks, h0, ranks)
                for idx in range(len(chunks)):
                    assert_allclose(self, ref_starts[idx], got_starts[idx])
                    assert_allclose(self, ref_vnew[idx], got_vnew[idx])
                assert_allclose(self, ref_final, got_final)

    def test_zero_initial_state(self):
        rng = np.random.default_rng(13)
        chunks = make_case(rng, num_chunks=3, chunk_size=64, seqlen=64 * 3 - 1)
        k_dim, v_dim = chunks[0]["k"].shape[1], chunks[0]["u"].shape[1]
        h0 = np.zeros((k_dim, v_dim))
        ref_starts, ref_vnew, ref_final = serial_run(chunks, h0)
        got_starts, got_vnew, got_final = cp_run(chunks, h0, rank_count(4, 3))
        for idx in range(len(chunks)):
            assert_allclose(self, ref_starts[idx], got_starts[idx])
            assert_allclose(self, ref_vnew[idx], got_vnew[idx])
        assert_allclose(self, ref_final, got_final)

    def test_fp32_error_model(self):
        """Estimate the FP32 gap against the FP64 golden before the device run.

        Simulates the kernel's storage/cast points: X/M/S summaries in FP32,
        h carried at float32 per chunk. Both the FP64 golden and the CP result
        are then rounded to bf16, mirroring the PTA comparison where the CPU
        reference is cast to the output dtype as well. The numbers are printed
        for the validation record.
        """
        def to_bf16(x):
            x32 = np.asarray(x, dtype=np.float32)
            u = x32.view(np.uint32)
            lsb = (u >> 16) & 1
            u = (u + (0x7FFF + lsb)) & 0xFFFF0000
            return u.view(np.float32)

        rng = np.random.default_rng(17)
        chunk_size, num_chunks = 64, 32
        chunks = make_case(rng, num_chunks=num_chunks, chunk_size=chunk_size,
                           seqlen=chunk_size * num_chunks, k_dim=64, v_dim=64)
        h0 = rng.standard_normal((64, 64)) * 0.1
        ref_starts, ref_vnew, ref_final = serial_run(chunks, h0)

        # FP32 summaries + FP32 prefix, replay states cast to float32 per chunk.
        f32 = np.float32
        summaries = []
        for begin, end in partition(num_chunks, 8):
            state = np.concatenate([np.eye(64, dtype=f32),
                                    np.zeros((64, 64), dtype=f32)], axis=1)
            for chunk in chunks[begin:end]:
                g = chunk["g"]
                q = np.exp(g[-1] - g).astype(f32)
                decay = f32(np.exp(g[-1]))
                proj = (chunk["w"].astype(f32) @ state).astype(f32)
                v_new = (chunk["u"].astype(f32) - proj[:, 64:]).astype(f32)
                rq = (-q[:, None] * proj[:, :64]).astype(f32)
                rv = (q[:, None] * v_new).astype(f32)
                delta = (chunk["k"].astype(f32).T @
                         np.concatenate([rq, rv], axis=1).astype(f32)).astype(f32)
                state = (decay * state + delta).astype(f32)
            summaries.append((state[:, :64].copy(), state[:, 64:].copy()))

        starts = [None] * num_chunks
        finals = None
        for rank, (begin, end) in enumerate(partition(num_chunks, 8)):
            b_trans = np.eye(64, dtype=f32)
            s_local = np.zeros((64, 64), dtype=f32)
            for prior in range(rank):
                b_bar, a_bar = summaries[prior]
                b_trans = (b_bar @ b_trans).astype(f32)
                s_local = (b_bar @ s_local + a_bar).astype(f32)
            state = (b_trans @ h0.astype(f32) + s_local).astype(f32)
            for idx in range(begin, end):
                starts[idx] = state.copy()
                state_f32 = state.astype(f32)
                state, _ = run_chunk(state_f32, chunks[idx])
                state = state.astype(f32)
            if rank == 7:
                finals = state

        def rel_stats(ref, got, quantize=True):
            # Same rule as tests/pta/data_compare_h.py: the absolute difference
            # is used when it is below the threshold, otherwise the relative
            # difference; 99.99% of elements must pass and max rel < 0.1.
            ref = np.asarray(to_bf16(ref) if quantize else ref, dtype=np.float64)
            got = np.asarray(to_bf16(got) if quantize else got, dtype=np.float64)
            absdiff = np.abs(ref - got)
            denom = np.maximum(np.abs(ref), np.abs(got))
            rel = np.where(absdiff < 1e-4, absdiff,
                           absdiff / np.maximum(denom, 1e-12))
            return float(np.max(rel)), float(np.mean(rel < 1e-4))

        for quantize, label in ((True, "bf16 out"), (False, "fp32 out")):
            for idx in (0, num_chunks // 2, num_chunks - 1):
                max_rel, pass_frac = rel_stats(ref_starts[idx], starts[idx], quantize)
                print("%s h[%d]: max rel %.3e, pass %.6f" %
                      (label, idx, max_rel, pass_frac))
            max_rel, pass_frac = rel_stats(ref_final, finals, quantize)
            print("%s final: max rel %.3e, pass %.6f" % (label, max_rel, pass_frac))
            self.assertLess(max_rel, 0.1)
            self.assertGreater(pass_frac, 0.9999)

    def test_log_scan_equals_linear_scan(self):
        """Future optimization: Hillis-Steele scan must match the linear one."""
        rng = np.random.default_rng(19)
        chunks = make_case(rng, num_chunks=9, chunk_size=32, seqlen=32 * 9)
        h0 = rng.standard_normal((chunks[0]["k"].shape[1],
                                  chunks[0]["u"].shape[1])) * 0.1
        ranks = 5
        slices = partition(len(chunks), ranks)
        summaries = [local_summary_projected(chunks[begin:end])
                     for begin, end in slices]
        linear = [prefix_state(summaries, rank, h0) for rank in range(ranks)]

        # Inclusive scan of affine pairs realised as (B, A) matrices.
        k_dim = h0.shape[0]
        v_dim = h0.shape[1]
        inc = [(b.copy(), a.copy()) for b, a in summaries]
        step = 1
        while step < ranks:
            nxt = [None] * ranks
            for rank in range(ranks):
                if rank >= step:
                    b_r, a_r = inc[rank]
                    b_p, a_p = inc[rank - step]
                    nxt[rank] = (b_r @ b_p, b_r @ a_p + a_r)
                else:
                    nxt[rank] = inc[rank]
            inc = nxt
            step *= 2
        log_scan = []
        for rank in range(ranks):
            b_inc, a_inc = inc[rank - 1] if rank > 0 else (np.eye(k_dim), np.zeros((k_dim, v_dim)))
            log_scan.append(b_inc @ h0 + a_inc)
        for expected, actual in zip(linear, log_scan):
            assert_allclose(self, expected, actual)


if __name__ == "__main__":
    unittest.main(verbosity=2)
