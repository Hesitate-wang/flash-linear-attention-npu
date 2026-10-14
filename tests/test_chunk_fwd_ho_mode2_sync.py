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
        state.run(events(self.ov2[:loop]), pingpongFlag=0)
        ping = 0
        for _ in range((rows + 31) // 32):
            state.run(events(body), pingpongFlag=ping)
            ping = 1 - ping
        state.run(events(self.ov2[end:]), pingpongFlag=ping)

    def test_event_domains_and_closed_fences(self):
        for text in (self.hv1, self.hv2):
            self.assertNotIn("EVENT_ID", text)
            self.assertTrue(all(4 <= expression(e, self.values) <= 7 for _, _, e in events(text)))
        for text in (self.ov1, self.ov2):
            self.assertTrue(all(0 <= expression(e, self.values, {"pingpongFlag": ping}) <= 3
                                for _, _, e in events(text) for ping in (0, 1)))
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
                state.run(self.fence, EventId=3)
                # OV1 processes both lanes even for a lane with zero token rows.
                state.run(self.ov1_trace(rows), pingpongFlag=0)
                state.run(self.fence, EventId=3)
                state.run(hv1)
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
        # Reproduce the original H/O collision using O's actual entry preset.
        loop = self.ov2.index("for (uint32_t row")
        state = EventState(self.values)
        state.run([("SetFlag", "MTE3_V", "0")])
        with self.assertRaisesRegex(AssertionError, "duplicate Set"):
            state.run(events(self.ov2[:loop]), pingpongFlag=0)
        state = EventState(self.values)
        with self.assertRaisesRegex(AssertionError, "Wait without producer"):
            state.run([("WaitFlag", "MTE3_V", "4")])
        preset, _ = self.driver_events()
        state.run(preset)
        with self.assertRaisesRegex(AssertionError, "Wait without producer"):
            state.run(events(self.ov2[loop:]), pingpongFlag=0)

    def test_stage_boundaries_and_direct_slot_release(self):
        ordered = ("WriteInitialHandoffState(", "while (scheduler.isRunning)",
                   "FenceOToH();", "RebuildCausalMask(", "epilogueQkmask(",
                   "FenceOToH();", "StageV1(h);", "StageV2(h, needStage2);",
                   "FenceHToO();", "epilogueOutput.ProcessNarrowSplit(",
                   "FenceOToH();", "Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(ov2ChunkFinished)")
        cursor = 0
        for marker in ordered:
            cursor = self.driver.index(marker, cursor) + len(marker)
        post = self.hv1.index("Vec1PostVF(")
        release = self.hv1.index("Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(hv1VUpdateReadyForHc2)")
        self.assertLess(post, release)
        self.assertIn("WaitFlag<AscendC::HardEvent::V_MTE3>", self.hv1[post:release])
        self.assertNotIn("SyncAll", self.driver)

    def test_ub_intervals_and_lifetimes(self):
        def region(suffix, size_suffix=None):
            return (self.value("UB_" + suffix + "_OFFSET"),
                    self.value("UB_" + suffix + "_OFFSET") + self.value("UB_" + (size_suffix or suffix) + "_BYTES"))

        def overlap(a, b):
            return max(a[0], b[0]) < min(a[1], b[1])

        direct = region("H_DIRECT")
        # Only OV1 is concurrent with HC1/HC2 writes. Its aliases stay outside.
        for name in ("O_BASE", "O_V1_A", "O_V1_OUT", "O_V1_OUT_HALF"):
            self.assertFalse(overlap(direct, region(name)), name)
        for name, size in (("O_V2_VWORK_PONG", "O_V2_VWORK"),
                           ("O_V2_HWORK_PONG", "O_V2_HWORK")):
            self.assertTrue(overlap(direct, region(name, size)))
        views = [region("O_V2_OUT"), region("O_V2_OUT_HALF"),
                 region("O_V2_OUT_PONG", "O_V2_OUT"),
                 region("O_V2_OUT_HALF_PONG", "O_V2_OUT_HALF")]
        for a, b in itertools.combinations(views, 2):
            self.assertFalse(overlap(a, b))
        self.assertEqual(max(end for _, end in views), self.value("UB_USED_MAX"))
        self.assertEqual(self.value("UB_USED_MAX"), 247 * 1024)
        self.assertLess(self.value("UB_USED_MAX"), 248 * 1024)
        for tokens in (1, 31, 32, 33, 63, 64, 65, 127, 128):
            for lane in (0, 1):
                rows = (tokens + 1) // 2 if lane == 0 else tokens // 2
                self.assertLessEqual(rows * 128 * 4, 32 * 1024)
                # Each 32-row output tile fits both ping/pong scratch formats.
                self.assertLessEqual(min(rows, 32) * 128 * 4, self.value("UB_O_V2_OUT_BYTES"))
                self.assertLessEqual(min(rows, 32) * 128 * 2, self.value("UB_O_V2_OUT_HALF_BYTES"))
        # H's active update output is at staging+48 KiB, not legacy 199 KiB.
        update = source(BLOCK / "block_epilogue_gdn_fwdh_update_preload_ss.hpp")
        self.assertIn("H_STAGING_OFFSET + 48U * 1024U", update)
        self.assertNotIn("H_UPDATE_SCRATCH_OFFSET", update)


if __name__ == "__main__":
    unittest.main()
