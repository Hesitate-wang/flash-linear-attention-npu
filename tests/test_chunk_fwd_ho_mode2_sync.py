"""Offline MODE2 event/UB contracts; not a CANN execution or precision test.

Run: python3 -m unittest discover -s tests -p test_chunk_fwd_ho_mode2_sync.py -v
"""

import itertools
from functools import lru_cache
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OP = ROOT / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h_o_fused"
ARCH = OP / "op_kernel/arch35"
BLOCK = ARCH / "epilogue/block"
PREFIX = "CHUNK_FWD_HO_FUSED_"
EVENT_RE = re.compile(
    r"AscendC::(SetFlag|WaitFlag)<AscendC::HardEvent::(\w+)>\(([^)]+)\)"
)


def source(path):
    text = path.read_text()
    return re.sub(r"/\*[\s\S]*?\*/|//[^\n]*", "", text)


def braced(text, start):
    begin = text.index("{", start)
    depth = 1
    end = begin + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[begin + 1:end - 1], end


def method(text, name):
    return braced(text, text.index(name))[0]


def events(text):
    return EVENT_RE.findall(text)


def select_if(text, condition, keep_body):
    pattern = re.compile(r"if\s*\(\s*" + re.escape(condition) + r"\s*\)")
    while True:
        match = pattern.search(text)
        if not match:
            return text
        body, end = braced(text, match.end())
        tail = text[end:]
        otherwise = re.match(r"\s*else\s*", tail)
        other_body = ""
        tail_end = end
        if otherwise:
            other_body, other_end = braced(tail, otherwise.end())
            tail_end = end + other_end
        selected = body if keep_body else other_body
        text = text[:match.start()] + selected + text[tail_end:]


def expression(expr, constants, variables=None):
    names = dict(constants)
    names.update(variables or {})
    expr = expr.replace("GDN::", "")
    expr = re.sub(r"\b(\d+)U\b", r"\1", expr)
    expr = re.sub(r"\bEVENT_ID(\d+)\b", r"\1", expr)
    expr = re.sub(r"\b[A-Za-z_]\w*\b", lambda m: str(names[m[0]]), expr)
    # The layout contract contains only integer constants, sums and products.
    if not re.fullmatch(r"[\d\s+*()\-]+", expr):
        raise AssertionError(f"unsupported integer expression: {expr}")
    return eval(" ".join(expr.split()), {"__builtins__": {}}, {})


@lru_cache(maxsize=1)
def constants():
    result = {}
    text = source(ARCH / "chunk_fwd_ho_fused_a5_single_slot_ub_layout.h")
    for name, value in re.findall(r"constexpr uint(?:16|32)_t (\w+)\s*=([^;]+);", text):
        result[name] = expression(value, result)
    return result


def conditional_paths(text):
    """Project local event traces through braced if/else paths, fail on new syntax.

    Conditions are intentionally nondeterministic: this checks a superset of
    dtype/state combinations rather than silently omitting a compile-time arm.
    Only used on H epilogues/initialization, whose event paths have no loops.
    """
    match = re.search(r"\bif\s*(?:constexpr\s*)?\(", text)
    if not match:
        return [events(text)]
    head = events(text[:match.start()])
    body, end = braced(text, match.end())
    arms = conditional_paths(body)
    tail = text[end:]
    otherwise = re.match(r"\s*else\b", tail)
    if otherwise:
        after_else = tail[otherwise.end():]
        if re.match(r"\s*if\b", after_else):
            # Gate dtype else-if arms contain no events in these methods.
            if events(body):
                raise AssertionError("eventful else-if requires an explicit model")
            return [head + rest for rest in conditional_paths(after_else)]
        other, end_other = braced(after_else, 0)
        arms += conditional_paths(other)
        tail = after_else[end_other:]
    else:
        arms += [[]]
    return [head + arm + rest for arm in arms for rest in conditional_paths(tail)]


@lru_cache(maxsize=256)
def resolved_trace(trace, variables):
    return tuple((action, direction, expression(expr, constants(), dict(variables)))
                 for action, direction, expr in trace)


class EventState:
    def __init__(self, values):
        self.values = values
        self.ready = set()

    def run(self, trace, **variables):
        for action, direction, event_id in resolved_trace(tuple(trace), tuple(sorted(variables.items()))):
            key = (direction, event_id)
            if action == "SetFlag":
                if key in self.ready:
                    raise AssertionError(f"duplicate Set without Wait: {key}")
                self.ready.add(key)
            else:
                if key not in self.ready:
                    raise AssertionError(f"Wait without producer: {key}")
                self.ready.remove(key)


class Mode2SyncTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values = constants()
        cls.kernel = source(ARCH / "gemm/kernel/gdn_fwd_ho_fused_kernel.hpp")
        cls.hv1 = method(source(BLOCK / "block_epilogue_gdn_fwdh_vnew_preload_ss.hpp"), "void operator()")
        cls.hv2 = method(source(BLOCK / "block_epilogue_gdn_fwdh_update_preload_ss.hpp"), "void operator()")
        cls.ov1 = method(source(BLOCK / "block_epilogue_gdn_fwdo_qkmask.hpp"), "void operator()")
        cls.ov2 = method(source(BLOCK / "block_epilogue_gdn_fwdo_output.hpp"), "void ProcessNarrowSplit(")
        cls.aic = method(cls.kernel, "void ProcessAic()")
        cls.driver = method(cls.kernel, "void ProcessAiv()")
        cls.fence = events(method(cls.kernel, "void FenceVectorUbReuse()"))

    def value(self, suffix):
        return self.values[PREFIX + suffix]

    def persistent(self):
        return {("MTE3_V", self.value("EVENT_H_OUTPUT")),
                ("MTE3_MTE2", self.value("EVENT_H_INPUT"))}

    def driver_events(self):
        before, loop = self.driver.split("while (scheduler.isRunning)", 1)
        _, end = braced(loop, 0)
        return events(before), events(loop[end:])

    @lru_cache(maxsize=8)
    def ov1_trace(self, rows):
        # Locate the row-count branch, not the earlier dtype/stride branches.
        row_branch = self.ov1.index("if(mActualThisSubBlock <= 32)")
        small, end = braced(self.ov1, row_branch)
        wide, _ = braced(self.ov1, self.ov1.index("else", end))
        text = small if rows <= 32 else wide
        prefix, output = text.split("AscendC::Mul(outUbTensor", 1) if rows <= 32 else ("", "")
        if rows <= 32:
            # Two dtype arms have the same event trace; execute exactly one.
            output = output[:output.index("else")]
            return events(prefix) + events(output)
        prefix, tile = re.split(r"for\s*\(uint32_t stage", wide, maxsplit=1)
        tile, _ = braced(tile, 0)
        tile = tile[:tile.index("if(std::is_same<AElementOutput")]
        cast_arm = wide[wide.index("if(std::is_same<AElementOutput"):]
        cast_arm, _ = braced(cast_arm, 0)
        stage_events = events(tile) + events(cast_arm)
        return events(prefix) + [(action, direction, expr.replace("pingpongFlag", str(ping)))
                                 for ping in (0, 1) for action, direction, expr in stage_events]

    def run_ov2(self, state, rows):
        if rows == 0:
            return  # The empty-row path only consumes cross-core flags.
        loop = self.ov2.index("for (uint32_t row")
        body, end = braced(self.ov2, loop)
        prefix = select_if(self.ov2[:loop], "!singleUbBuffer", False)
        prefix = select_if(prefix, "singleUbBuffer", True)
        body = select_if(body, "!singleUbBuffer", False)
        body = select_if(body, "singleUbBuffer", True)
        suffix = select_if(self.ov2[end:], "!singleUbBuffer", False)
        suffix = select_if(suffix, "singleUbBuffer", True)
        state.run(events(prefix), pingpongFlag=0, bufferIndex=0)
        for _ in range((rows + 31) // 32):
            state.run(events(body), pingpongFlag=0, bufferIndex=0)
        state.run(events(suffix), pingpongFlag=0, bufferIndex=0)

    def test_event_domains_and_closed_fences(self):
        for text in (self.hv1, self.hv2):
            self.assertNotIn("EVENT_ID", text)
            self.assertTrue(all(4 <= expression(e, self.values) <= 7 for _, _, e in events(text)))
        for text in (self.ov1, self.ov2):
            self.assertTrue(all(0 <= expression(
                e, self.values, {"pingpongFlag": ping, "bufferIndex": index}) <= 3
                for _, _, e in events(text)
                for ping, index in itertools.product((0, 1), repeat=2)))
        for event_id in (3, 7):
            state = EventState(self.values)
            state.ready = self.persistent().copy()
            state.run(self.fence, EventId=event_id)
            self.assertEqual(state.ready, self.persistent())
        self.assertEqual([direction for action, direction, _ in self.fence if action == "SetFlag"],
                         ["MTE2_V", "V_MTE3", "MTE3_V", "MTE3_MTE2"])

    def test_initialization_does_not_leave_tokens(self):
        init = method(self.kernel, "void WriteInitialHandoffState(")
        init = init[init.index("if (useInitialState)"):]
        for trace in conditional_paths(init):
            state = EventState(self.values)
            state.run(trace)
            self.assertFalse(state.ready)
            self.assertTrue(all(expression(e, self.values) == 5 for _, _, e in trace))

    def test_multichunk_state_and_empty_rows(self):
        preset, drain = self.driver_events()
        self.assertEqual(len(preset), 2)
        self.assertEqual(len(drain), 2)
        hv1_paths = set(map(tuple, conditional_paths(self.hv1)))
        hv2_paths = set(map(tuple, conditional_paths(self.hv2)))
        for rows, chunks, store_final, hv1, hv2 in itertools.product(
                (0, 1, 16, 17, 31, 32, 33, 64), (0, 1, 3, 16), (False, True), hv1_paths, hv2_paths):
            state = EventState(self.values)
            state.run(preset)
            for chunk in range(chunks):
                # OV1 processes both lanes even for a lane with zero token rows.
                state.run(self.ov1_trace(rows), pingpongFlag=0)
                state.run(hv1)
                # OV1 QK output is reused by HV2, not HV1.
                state.run(self.fence, EventId=3)
                if store_final or chunk + 1 < chunks:
                    state.run(hv2)
                state.run(self.fence, EventId=7)
                self.run_ov2(state, rows)
                state.run(self.fence, EventId=3)
                self.assertEqual(state.ready, self.persistent())
            state.run(drain)
            self.assertFalse(state.ready)

    def test_model_rejects_collision_missing_wait_and_missing_producer(self):
        state = EventState(self.values)
        state.run([("SetFlag", "MTE3_V", "0")])
        with self.assertRaisesRegex(AssertionError, "duplicate Set"):
            state.run([("SetFlag", "MTE3_V", "0")])
        # The serialized MODE2 output path requires a producer before reusing
        # its single MTE3_V slot.
        loop = self.ov2.index("for (uint32_t row")
        state = EventState(self.values)
        with self.assertRaisesRegex(AssertionError, "Wait without producer"):
            state.run([("WaitFlag", "MTE3_V", "4")])
        with self.assertRaisesRegex(AssertionError, "Wait without producer"):
            single_path = select_if(self.ov2[loop:], "!singleUbBuffer", False)
            single_path = select_if(single_path, "singleUbBuffer", True)
            state.run(events(single_path), pingpongFlag=0, bufferIndex=0)

    def test_stage_boundaries_and_direct_slot_release(self):
        ordered = ("WriteInitialHandoffState(", "RebuildCausalMask(",
                   "while (scheduler.isRunning)", "epilogueQkmask(",
                   "StageV1(h);", "FenceOToH();", "StageV2(h, needStage2);",
                   "FenceHToO();", "epilogueOutput.ProcessNarrowSplit(",
                   "FenceOToH();", "Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(ov2ChunkFinished)")
        cursor = 0
        for marker in ordered:
            cursor = self.driver.index(marker, cursor) + len(marker)
        self.assertEqual(self.driver.count("RebuildCausalMask("), 1)
        self.assertLess(self.driver.index("RebuildCausalMask("),
                        self.driver.index("while (scheduler.isRunning)"))
        post = self.hv1.index("Vec1PostVF(")
        release = self.hv1.index("Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(hv1VUpdateReadyForHc2)")
        self.assertLess(post, release)
        self.assertIn("WaitFlag<AscendC::HardEvent::V_MTE3>", self.hv1[post:release])
        self.assertNotIn("SyncAll", self.driver)

    def test_aic_chunk_gate_precedes_all_chunk_work(self):
        loop = self.aic.index("while (scheduler.isRunning)")
        gate = self.aic.index("Arch::CrossCoreWaitFlag(ov2ChunkFinished);", loop)
        oc1 = self.aic.index("blockMmadQK(tileQ, tileK, tileAttn, shape);", loop)
        hc1 = self.aic.index("StageC1(h);", loop)
        oc2 = self.aic.index("blockMmadQH(tileQ, tileH, tileHWork, shape);", loop)
        self.assertLess(gate, oc1)
        self.assertLess(oc1, hc1)
        self.assertLess(hc1, oc2)
        self.assertEqual(self.aic[loop:].count("CrossCoreWaitFlag(ov2ChunkFinished)"), 1)

    def test_tail_does_not_publish_next_chunk_tokens(self):
        for flag in ("hv2HReadyForHc1", "ov2ChunkFinished"):
            set_call = f"Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>({flag});"
            set_pos = self.driver.rindex(set_call)
            guard_pos = self.driver.rfind("if (!scheduler.IsFinalChunk())", 0, set_pos)
            self.assertGreaterEqual(guard_pos, 0, f"{flag} must only be set when a next chunk exists")
            self.assertLess(set_pos - guard_pos, 100)

        # Both tokens are seeded for chunk 0. Each chunk consumes one
        # generation; only non-final chunks publish the next generation.
        for chunk_count in (1, 2, 128):
            pending = {"hv2HReadyForHc1": 1, "ov2ChunkFinished": 1}
            for chunk_idx in range(chunk_count):
                for flag in pending:
                    self.assertEqual(pending[flag], 1)
                    pending[flag] -= 1
                if chunk_idx + 1 < chunk_count:
                    for flag in pending:
                        pending[flag] += 1
            self.assertEqual(pending, {"hv2HReadyForHc1": 0, "ov2ChunkFinished": 0})

    def test_ub_intervals_and_lifetimes(self):
        def region(suffix, size_suffix=None, offset_suffix=None):
            size_suffix = size_suffix or suffix
            offset_suffix = offset_suffix or suffix
            return (self.value("UB_VAR_" + offset_suffix + "_OFFSET"),
                    self.value("UB_VAR_" + offset_suffix + "_OFFSET")
                    + self.value("UB_VAR_" + size_suffix + "_BYTES"))

        def overlap(a, b):
            return max(a[0], b[0]) < min(a[1], b[1])

        direct = region("H_DIRECT")
        qk_scratch = [region("QK_A"), region("QK_OUT"), region("QK_OUT_LOW")]
        for a, b in itertools.combinations(qk_scratch, 2):
            self.assertFalse(overlap(a, b))
        for view in qk_scratch:
            self.assertFalse(overlap(direct, view))
        work_a = region("O_WORK_A")
        work_h = region("O_WORK_H")
        out = region("O_OUT")
        out_low = region("O_OUT_LOW")
        self.assertFalse(overlap(work_a, work_h))
        self.assertFalse(overlap(work_h, out))
        self.assertFalse(overlap(out, out_low))
        self.assertEqual(out[0], 72 * 1024)
        self.assertEqual(out_low[1], 96 * 1024)
        self.assertEqual(self.value("UB_USED_MAX"), 248 * 1024)
        self.assertLessEqual(self.value("UB_USED_MAX"), 248 * 1024)
        self.assertNotEqual(self.value("UB_VAR_GATE_OFFSET"), self.value("UB_VAR_G_LAST_OFFSET"))
        self.assertEqual(self.value("UB_VAR_GATE_OFFSET"), self.value("UB_VAR_O_GATE_OFFSET"))
        self.assertEqual(self.value("UB_VAR_GATE_INPUT_OFFSET"),
                         self.value("UB_VAR_O_GATE_INPUT_OFFSET"))
        init_state = region("INIT_STATE")
        init_h = region("INIT_H")
        calc = region("CALC")
        mask = region("MASK")
        g_last = region("G_LAST")
        self.assertFalse(overlap(init_state, init_h))
        self.assertFalse(overlap(init_state, calc))
        self.assertFalse(overlap(mask, calc))
        self.assertFalse(overlap(mask, direct))
        self.assertFalse(overlap(g_last, region("GATE")))
        h_vnew = source(BLOCK / "block_epilogue_gdn_fwdh_vnew_preload_ss.hpp")
        self.assertIn("GetBufferByByte<GElementInput>(SLOT_G_INPUT_BUF_OFFSET)", h_vnew)
        self.assertEqual(self.value("UB_VAR_O_WORK_END"), 165 * 1024)
        self.assertTrue(overlap(work_a, direct))
        self.assertTrue(overlap(work_h, direct))
        self.assertFalse(overlap(out, direct))
        update_scratch = region("H_UPDATE_SCRATCH")
        self.assertEqual(update_scratch, (71 * 1024, 103 * 1024))
        self.assertTrue(overlap(update_scratch, region("QK_A")))
        self.assertTrue(overlap(update_scratch, region("QK_OUT")))
        self.assertTrue(overlap(update_scratch, out))
        self.assertEqual(self.value("UB_VAR_H_DIRECT_END"), self.value("UB_VAR_GATE_OFFSET"))
        # QK-to-H and init-to-runtime aliases are separated by their stage order.
        self.assertLess(self.value("UB_VAR_QK_SCRATCH_END"), self.value("UB_VAR_H_DIRECT_OFFSET"))
        l1 = source(ARCH / "chunk_fwd_ho_fused_a5_single_slot_l1_layout.h")
        l1_values = {}
        for name, value in re.findall(r"constexpr uint32_t (\w+)\s*=\s*([^;]+);", l1):
            l1_values[name] = expression(value, l1_values)
        h_window = (l1_values["CHUNK_FWD_HO_FUSED_L1_H_STAGE_BASE"],
                    l1_values["CHUNK_FWD_HO_FUSED_L1_H_STAGE_BASE"] + 128 * 1024)
        oc1_window = (l1_values["CHUNK_FWD_HO_FUSED_L1_O_CUBE1_BASE"],
                      l1_values["CHUNK_FWD_HO_FUSED_L1_O_CUBE1_BASE"] + 128 * 1024)
        oc2_window = (l1_values["CHUNK_FWD_HO_FUSED_L1_O_CUBE2_BASE"],
                      l1_values["CHUNK_FWD_HO_FUSED_L1_O_CUBE2_BASE"] + 128 * 1024)
        oc3_window = (l1_values["CHUNK_FWD_HO_FUSED_L1_O_CUBE3_BASE"],
                      l1_values["CHUNK_FWD_HO_FUSED_L1_O_CUBE3_BASE"] + 128 * 1024)
        self.assertFalse(overlap(h_window, oc2_window))
        self.assertFalse(overlap(h_window, oc3_window))
        self.assertFalse(overlap(h_window, oc1_window))
        self.assertFalse(overlap(oc1_window, oc2_window))
        self.assertFalse(overlap(oc2_window, oc3_window))
        for tokens in (1, 31, 32, 33, 63, 64, 65, 127, 128):
            for lane in (0, 1):
                rows = (tokens + 1) // 2 if lane == 0 else tokens // 2
                self.assertLessEqual(rows * 128 * 4, 32 * 1024)
                self.assertLessEqual(min(rows, 32) * 128 * 4,
                                     self.value("UB_VAR_O_OUT_BYTES"))
                self.assertLessEqual(min(rows, 32) * 128 * 2,
                                     self.value("UB_VAR_O_OUT_LOW_BYTES"))
        kernel = self.kernel
        self.assertIn("EpilogueOQkmask epilogueQkmask(resource, true)", kernel)
        self.assertIn("EpilogueOOutput epilogueOutput(resource, true)", kernel)
        output = source(BLOCK / "block_epilogue_gdn_fwdo_output.hpp")
        self.assertIn("bool useFusedUbLayout = false", output)
        self.assertIn("if (useFusedUbLayout)", output)
        self.assertIn("aUbTensorPong = aUbTensorPing", output)
        self.assertIn("outUbTensorPong = outUbTensorPing", output)
        narrow = method(output, "void ProcessNarrowSplit(")
        self.assertLess(narrow.index("WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0)",
                                     narrow.index("for (uint32_t row")),
                        narrow.index("OutputAddScaleVf"))
        self.assertIn("if (singleUbBuffer)", narrow)
        self.assertIn("WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0)", narrow)
        # MODE1's default branch retains its original local address plan.
        mode1 = source(ARCH / "gemm/kernel/gdn_fwd_o_kernel.hpp")
        self.assertIn("EpilogueGDNFwdOQkmask epilogueGDNFwdOQkmask(resource)", mode1)


if __name__ == "__main__":
    unittest.main()
