import math
import random
import unittest


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
            for i in range(len(a))]


def add(a, b):
    return [[x + y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]


def make_chunk(rng, k, v, length, gate_mode):
    w = [[rng.uniform(-0.1, 0.1) for _ in range(k)] for _ in range(length)]
    key = [[rng.uniform(-0.1, 0.1) for _ in range(k)] for _ in range(length)]
    u = [[rng.uniform(-0.1, 0.1) for _ in range(v)] for _ in range(length)]
    g = [rng.uniform(-0.2, 0.0) for _ in range(length)] if gate_mode in ("g", "both") else None
    gk = [rng.uniform(-0.1, 0.0) for _ in range(k)] if gate_mode in ("gk", "both") else None
    g_last = g[-1] if g is not None else 0.0
    factor = [math.exp(g_last - x) for x in g] if g is not None else [1.0] * length
    decay = [math.exp(g_last) * (2.0 ** gk[i] if gk is not None else 1.0) for i in range(k)]

    a = [[(decay[i] if i == j else 0.0) -
          sum(key[t][i] * factor[t] * w[t][j] for t in range(length))
          for j in range(k)] for i in range(k)]
    e = [[sum(key[t][i] * factor[t] * u[t][j] for t in range(length))
          for j in range(v)] for i in range(k)]
    return {"a": a, "e": e, "w": w, "key": key, "u": u, "factor": factor, "decay": decay}


def build_chunk_affine_by_subblock(chunk, k, v):
    a = [[0.0 for _ in range(k)] for _ in range(k)]
    e = [[0.0 for _ in range(v)] for _ in range(k)]
    rows_per_subblock = k // 2
    cols_per_subblock = v // 2
    for subblock in range(2):
        row_begin = subblock * rows_per_subblock
        col_begin = subblock * cols_per_subblock
        for i in range(row_begin, row_begin + rows_per_subblock):
            for j in range(k):
                a[i][j] = (chunk["decay"][i] if i == j else 0.0) - sum(
                    chunk["key"][t][i] * chunk["factor"][t] * chunk["w"][t][j]
                    for t in range(len(chunk["w"]))
                )
        for i in range(k):
            for j in range(col_begin, col_begin + cols_per_subblock):
                e[i][j] = sum(
                    chunk["key"][t][i] * chunk["factor"][t] * chunk["u"][t][j]
                    for t in range(len(chunk["u"]))
                )
    return a, e


def run_chunk(h, chunk):
    vnew = [[chunk["u"][t][v] - sum(chunk["w"][t][i] * h[i][v] for i in range(len(h)))
             for v in range(len(h[0]))] for t in range(len(chunk["u"]))]
    delta = [[sum(chunk["key"][t][i] * chunk["factor"][t] * vnew[t][v]
                  for t in range(len(vnew))) for v in range(len(h[0]))] for i in range(len(h))]
    h_next = [[chunk["decay"][i] * h[i][v] + delta[i][v] for v in range(len(h[0]))]
              for i in range(len(h))]
    return h_next, vnew


def summarize(chunks, k, v):
    m = [[float(i == j) for j in range(k)] for i in range(k)]
    e = [[0.0 for _ in range(v)] for _ in range(k)]
    for chunk in chunks:
        m = matmul(chunk["a"], m)
        e = add(matmul(chunk["a"], e), chunk["e"])
    return m, e


def partition(length, ranks):
    base, rem = divmod(length, ranks)
    start = 0
    result = []
    for rank in range(ranks):
        count = base + (rank < rem)
        result.append((start, start + count))
        start += count
    return result


class CpAffineTest(unittest.TestCase):
    def test_two_aiv_subblocks_cover_value_axis_once(self):
        sub_block_count = 2
        value_dim = 128
        owned = []
        for sub_block in range(sub_block_count):
            width = value_dim // sub_block_count
            owned.extend(range(sub_block * width, (sub_block + 1) * width))
        self.assertEqual(owned, list(range(value_dim)))

    def test_rank_summaries_match_serial_chunks(self):
        k, v = 4, 3
        for gate_mode in ("g", "gk", "both"):
            rng = random.Random(100 + len(gate_mode))
            chunks = [make_chunk(rng, k, v, length, gate_mode) for length in (32, 32, 17, 32, 5)]
            initial = [[rng.uniform(-0.2, 0.2) for _ in range(v)] for _ in range(k)]
            ref_h = [row[:] for row in initial]
            ref_starts, ref_vnew = [], []
            for chunk in chunks:
                ref_starts.append([row[:] for row in ref_h])
                ref_h, vnew = run_chunk(ref_h, chunk)
                ref_vnew.append(vnew)

            for ranks in (1, 2, 3):
                slices = partition(len(chunks), ranks)
                summaries = [summarize(chunks[begin:end], k, v) for begin, end in slices]
                got_h, got_vnew = [None] * len(chunks), [None] * len(chunks)
                got_final = None
                for rank, (begin, end) in enumerate(slices):
                    incoming = [row[:] for row in initial]
                    for prior in range(rank):
                        pm, pe = summaries[prior]
                        incoming = add(matmul(pm, incoming), pe)
                    for idx in range(begin, end):
                        got_h[idx] = [row[:] for row in incoming]
                        incoming, got_vnew[idx] = run_chunk(incoming, chunks[idx])
                    if rank == ranks - 1:
                        got_final = incoming

                for expected, actual in zip(ref_starts, got_h):
                    for erow, arow in zip(expected, actual):
                        for x, y in zip(erow, arow):
                            self.assertAlmostEqual(x, y, places=11)
                for expected, actual in zip(ref_vnew, got_vnew):
                    for erow, arow in zip(expected, actual):
                        for x, y in zip(erow, arow):
                            self.assertAlmostEqual(x, y, places=11)
                for erow, arow in zip(ref_h, got_final):
                    for x, y in zip(erow, arow):
                        self.assertAlmostEqual(x, y, places=11)

    def test_subblock_affine_build_matches_full_chunk_formula(self):
        k, v = 8, 6
        for gate_mode in ("g", "gk", "both"):
            for length in (32, 17):
                rng = random.Random(400 + length + len(gate_mode))
                chunk = make_chunk(rng, k, v, length, gate_mode)
                got_a, got_e = build_chunk_affine_by_subblock(chunk, k, v)
                for expected, actual in ((chunk["a"], got_a), (chunk["e"], got_e)):
                    for expected_row, actual_row in zip(expected, actual):
                        for x, y in zip(expected_row, actual_row):
                            self.assertAlmostEqual(x, y, places=12)


if __name__ == "__main__":
    unittest.main()
