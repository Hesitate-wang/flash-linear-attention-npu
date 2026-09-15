# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import ctypes
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ASCENDC_DIR = Path(__file__).resolve().parents[1] / "fla_npu" / "ops" / "ascendc"


def load_aclnn_ctypes_module():
    package_name = "fla_npu_test_aclnn_ctypes"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ASCENDC_DIR)]
    sys.modules[package_name] = package

    for module_name in ("_runtime", "_kda_policy", "_aclnn_ctypes"):
        qualified_name = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(
            qualified_name,
            ASCENDC_DIR / f"{module_name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)

    return sys.modules[f"{package_name}._aclnn_ctypes"]


ACLNN_CTYPES = load_aclnn_ctypes_module()


class FakeTensor:
    def __init__(self, shape, dtype=None, *, device_type="npu", contiguous=True):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype
        self.device = types.SimpleNamespace(type=device_type)
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous

    def storage_offset(self):
        return 0


class FakeCallContext:
    def __init__(self):
        self.descriptor_names = []
        self.descriptor_metadata = []

    def tensor(
        self,
        tensor,
        name,
        *,
        acl_format_override=None,
        storage_shape_override=None,
    ):
        self.descriptor_names.append(name)
        self.descriptor_metadata.append(
            (name, tensor, acl_format_override, storage_shape_override)
        )
        return ctypes.c_void_p(0x1000 + len(self.descriptor_names))

    def int_array(self, values):
        del values
        self.descriptor_names.append("query_start_loc")
        return ctypes.c_void_p(0x2000)


class AclnnCtypesAbiTest(unittest.TestCase):
    def test_chunk_fwd_h_o_fused_signature_and_output_contract(self):
        expected_argtypes = [
            *([ctypes.c_void_p] * 9),
            ctypes.c_bool,
            ctypes.c_int64,
            ctypes.c_double,
            ctypes.c_bool,
            ctypes.c_bool,
            ctypes.c_char_p,
            *([ctypes.c_void_p] * 2),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.assertEqual(
            ACLNN_CTYPES._GET_WORKSPACE_ARGTYPES["aclnnChunkFwdHOFused"],
            expected_argtypes,
        )

        fake_torch = types.ModuleType("torch")
        fake_torch.float16 = object()
        fake_torch.bfloat16 = object()
        fake_torch.float32 = object()
        dtype = fake_torch.bfloat16
        k = FakeTensor((1, 2, 129, 128), dtype)
        q = FakeTensor(k.shape, dtype)
        w = FakeTensor((1, 4, 129, 128), dtype)
        u = FakeTensor((1, 4, 129, 256), dtype)
        g = FakeTensor((1, 4, 129), fake_torch.float32)
        captured = {}

        def fake_empty(shape, like, **kwargs):
            return FakeTensor(shape, kwargs.get("dtype", like.dtype))

        def fake_call_aclnn(name, build_args, outputs):
            context = FakeCallContext()
            captured["name"] = name
            captured["args"] = build_args(context)
            captured["descriptors"] = context.descriptor_names
            return outputs

        with mock.patch.dict(sys.modules, {"torch": fake_torch, "torch_npu": None}), \
                mock.patch.object(ACLNN_CTYPES, "_empty", side_effect=fake_empty), \
                mock.patch.object(ACLNN_CTYPES, "_acl_format", return_value=ACLNN_CTYPES.ACL_FORMAT_ND), \
                mock.patch.object(ACLNN_CTYPES, "_call_aclnn", side_effect=fake_call_aclnn):
            outputs = ACLNN_CTYPES.npu_chunk_fwd_h_o_fused(
                k,
                w,
                u,
                g,
                q,
                output_final_state=True,
                chunk_size=64,
                scale=0.125,
            )

        self.assertEqual(captured["name"], "aclnnChunkFwdHOFused")
        self.assertEqual(len(captured["args"]), 17)
        self.assertEqual(captured["descriptors"][:7], ["k", "w", "u", "g", "gk", "initial_state", "q"])
        self.assertEqual(captured["descriptors"][-2:], ["o", "final_state"])
        self.assertEqual([output.shape for output in outputs], [
            (1, 4, 129, 256),
            (1, 4, 128, 256),
        ])
        self.assertIs(outputs[1].dtype, fake_torch.float32)

    def test_gdn_training_and_inference_output_contract(self):
        import inspect

        function = ACLNN_CTYPES.npu_chunk_gated_delta_rule_fwd
        self.assertIs(inspect.signature(function).parameters["disable_recompute"].default, True)
        fake_torch = types.ModuleType("torch")
        fake_torch.float32 = object()
        fake_torch.bfloat16 = object()
        q = FakeTensor((1, 2, 65, 128), fake_torch.bfloat16)
        v = FakeTensor((1, 4, 65, 256), fake_torch.bfloat16)
        g = FakeTensor((1, 65, 4), fake_torch.float32)
        beta = FakeTensor(g.shape, fake_torch.bfloat16)
        state = FakeTensor((1, 4, 128, 256), fake_torch.float32)
        captured = {}

        def fake_empty(shape, like, **kwargs):
            return FakeTensor(shape, kwargs.get("dtype", like.dtype))

        def fake_call_aclnn(name, build_args, outputs):
            context = FakeCallContext()
            captured["name"] = name
            captured["args"] = build_args(context)
            captured["tensors"] = {row[0]: row[1] for row in context.descriptor_metadata}
            return outputs

        modes = (("default", {}, True), ("none", {"disable_recompute": None}, True),
                 ("training", {"disable_recompute": True}, True),
                 ("inference", {"disable_recompute": False}, False))
        with mock.patch.dict(sys.modules, {"torch": fake_torch}), \
                mock.patch.object(ACLNN_CTYPES, "_empty", side_effect=fake_empty), \
                mock.patch.object(ACLNN_CTYPES, "_call_aclnn", side_effect=fake_call_aclnn):
            for mode, options, training in modes:
                for with_h in (False, True):
                    for with_final_state in (False, True):
                        with self.subTest(mode=mode, h=with_h, final_state=with_final_state):
                            outputs = function(
                                q, q, v, g, beta, initial_state=state,
                                output_final_state=with_final_state,
                                return_intermediate_states=with_h, **options,
                            )
                            tensors = captured["tensors"]
                            self.assertEqual(captured["name"], "aclnnChunkGatedDeltaRuleFwd")
                            self.assertEqual(len(captured["args"]), 27)
                            self.assertEqual(len(outputs), (4 if training else 2) + int(with_h))
                            self.assertIs(outputs[0], tensors["o"])
                            self.assertEqual(outputs[0].shape, (1, 65, 4, 256))
                            self.assertIs(outputs[1], tensors["final_state"])
                            self.assertEqual(outputs[1] is not None, with_final_state)
                            if training:
                                self.assertIs(outputs[2], tensors["g_cumsum"])
                                self.assertIs(outputs[3], tensors["A"])
                                self.assertEqual(outputs[2].shape, (1, 65, 4))
                                self.assertIs(outputs[2].dtype, fake_torch.float32)
                                self.assertEqual(outputs[3].shape, (1, 4, 65, 64))
                            else:
                                self.assertIsNone(tensors["g_cumsum"])
                                self.assertIsNone(tensors["A"])
                            if with_h:
                                self.assertIs(outputs[-1], tensors["h"])
                                self.assertEqual(outputs[-1].shape, (1, 4, 2, 128, 256))
                            else:
                                self.assertIsNone(tensors["h"])

    def test_chunk_gdn_bwd_intra_signature_has_no_debug_stage(self):
        import inspect

        expected_argtypes = [
            *([ctypes.c_void_p] * 9),
            ctypes.c_double,
            ctypes.c_int64,
            ctypes.c_bool,
            *([ctypes.c_void_p] * 3),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.assertEqual(
            ACLNN_CTYPES._GET_WORKSPACE_ARGTYPES["aclnnChunkGdnBwdIntra"],
            expected_argtypes,
        )
        signature = inspect.signature(ACLNN_CTYPES.npu_chunk_gdn_bwd_intra)
        self.assertNotIn("stage", signature.parameters)
        self.assertIs(signature.parameters["use_exp2"].default, True)

        captured = {}
        fake_torch = types.ModuleType("torch")
        fake_torch.Tensor = FakeTensor
        fake_torch.float16 = object()
        fake_torch.bfloat16 = object()
        fake_torch.float32 = object()

        def fake_empty(shape, like, **kwargs):
            return FakeTensor(shape, kwargs.get("dtype", like.dtype))

        def fake_empty_like(tensor, **kwargs):
            return FakeTensor(tensor.shape, kwargs.get("dtype", tensor.dtype))

        def fake_call_aclnn(name, build_args, outputs):
            context = FakeCallContext()
            captured["name"] = name
            captured["args"] = build_args(context)
            return outputs

        dtype = fake_torch.bfloat16
        q = FakeTensor((1, 3, 65, 128), dtype)
        k = FakeTensor(q.shape, dtype)
        v = FakeTensor((1, 6, 65, 128), dtype)
        g = FakeTensor((1, 6, 65), fake_torch.float32)
        beta = FakeTensor((1, 6, 65), fake_torch.bfloat16)
        a = FakeTensor((1, 6, 65, 64), dtype)
        d_o = FakeTensor(v.shape, dtype)

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            with mock.patch.object(ACLNN_CTYPES, "_empty", side_effect=fake_empty):
                with mock.patch.object(
                    ACLNN_CTYPES, "_empty_like", side_effect=fake_empty_like
                ):
                    with mock.patch.object(
                        ACLNN_CTYPES, "_call_aclnn", side_effect=fake_call_aclnn
                    ):
                        outputs = ACLNN_CTYPES.npu_chunk_gdn_bwd_intra(
                            q, k, v, g, beta, a, d_o, 0.125, 64
                        )

        operator_argtypes = expected_argtypes[:-2]
        self.assertEqual(captured["name"], "aclnnChunkGdnBwdIntra")
        self.assertEqual(len(outputs), 3)
        self.assertEqual(len(captured["args"]), len(operator_argtypes))
        self.assertEqual([type(arg) for arg in captured["args"]], operator_argtypes)
        self.assertTrue(captured["args"][11].value)

    def test_chunk_gated_delta_rule_bwd_dhu_signature_and_default_use_exp2(self):
        import inspect

        import torch

        expected_argtypes = [
            *([ctypes.c_void_p] * 11),
            ctypes.c_double,
            ctypes.c_int64,
            ctypes.c_bool,
            ctypes.c_bool,
            *([ctypes.c_void_p] * 3),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.assertEqual(
            ACLNN_CTYPES._GET_WORKSPACE_ARGTYPES["aclnnChunkGatedDeltaRuleBwdDhu"],
            expected_argtypes,
        )
        self.assertIs(
            inspect.signature(ACLNN_CTYPES.npu_chunk_gated_delta_rule_bwd_dhu)
            .parameters["use_exp2"]
            .default,
            False,
        )

        captured = {}

        def fake_empty(shape, like, **kwargs):
            return FakeTensor(shape, kwargs.get("dtype", like.dtype))

        def fake_empty_like(tensor, **kwargs):
            return FakeTensor(tensor.shape, kwargs.get("dtype", tensor.dtype))

        def fake_call_aclnn(name, build_args, outputs):
            context = FakeCallContext()
            captured["name"] = name
            captured["args"] = build_args(context)
            return outputs

        dtype = torch.float16
        q = FakeTensor((1, 2, 64, 128), dtype)
        state = FakeTensor((1, 2, 64, 128), dtype)
        g = FakeTensor((1, 2, 64), torch.float32)
        with mock.patch.object(ACLNN_CTYPES, "_empty", side_effect=fake_empty):
            with mock.patch.object(ACLNN_CTYPES, "_empty_like", side_effect=fake_empty_like):
                with mock.patch.object(ACLNN_CTYPES, "_call_aclnn", side_effect=fake_call_aclnn):
                    ACLNN_CTYPES.npu_chunk_gated_delta_rule_bwd_dhu(
                        q, q, q, state, state, scale=0.125, chunk_size=64, g=g
                    )

        operator_argtypes = expected_argtypes[:-2]
        self.assertEqual(captured["name"], "aclnnChunkGatedDeltaRuleBwdDhu")
        self.assertEqual(len(captured["args"]), len(operator_argtypes))
        self.assertEqual([type(arg) for arg in captured["args"]], operator_argtypes)
        self.assertFalse(captured["args"][13].value)
        self.assertFalse(captured["args"][14].value)

    def test_recurrent_gated_delta_rule_requires_at_least_one_gate_before_launch(self):
        with mock.patch.object(ACLNN_CTYPES, "_call_aclnn") as call_aclnn:
            with self.assertRaisesRegex(
                RuntimeError,
                r"^npu_recurrent_gated_delta_rule: either g or gk must be provided\.$",
            ):
                ACLNN_CTYPES.npu_recurrent_gated_delta_rule(
                    None,
                    None,
                    None,
                    None,
                    beta=None,
                    actual_seq_lengths=None,
                    ssm_state_indices=None,
                )

        call_aclnn.assert_not_called()

    def test_recurrent_gated_delta_rule_signature_matches_aclnn_prototype(self):
        expected_argtypes = [
            *([ctypes.c_void_p] * 10),
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.assertEqual(
            ACLNN_CTYPES._GET_WORKSPACE_ARGTYPES["aclnnRecurrentGatedDeltaRule"],
            expected_argtypes,
        )

    def test_recurrent_gated_delta_rule_wrapper_uses_nd_descriptors(self):
        captured = {}
        fake_torch = types.ModuleType("torch")
        fake_torch.Tensor = FakeTensor
        fake_torch.bfloat16 = object()
        fake_torch.float32 = object()
        fake_torch.int32 = object()

        inputs = {
            "query": FakeTensor((5, 4, 128), fake_torch.bfloat16),
            "key": FakeTensor((5, 4, 128), fake_torch.bfloat16),
            "value": FakeTensor((5, 8, 128), fake_torch.bfloat16),
            "state": FakeTensor(
                (5, 8, 128, 128),
                fake_torch.float32,
                contiguous=False,
            ),
            "beta": FakeTensor((5, 8), fake_torch.bfloat16),
            "actual_seq_lengths": FakeTensor((3,), fake_torch.int32),
            "ssm_state_indices": FakeTensor((5,), fake_torch.int32),
            "g": FakeTensor((5, 8), fake_torch.float32),
            "gk": FakeTensor((5, 8, 128), fake_torch.float32),
            "num_accepted_tokens": FakeTensor((2,), fake_torch.int32),
        }

        def fake_empty(shape, like, **kwargs):
            return FakeTensor(shape, kwargs.get("dtype", like.dtype))

        def fake_call_aclnn(name, build_args, outputs):
            context = FakeCallContext()
            captured["name"] = name
            captured["args"] = build_args(context)
            captured["descriptor_names"] = context.descriptor_names
            captured["descriptor_metadata"] = context.descriptor_metadata
            return outputs

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            with mock.patch.object(ACLNN_CTYPES, "_empty", side_effect=fake_empty):
                with mock.patch.object(ACLNN_CTYPES, "_call_aclnn", side_effect=fake_call_aclnn):
                    output = ACLNN_CTYPES.npu_recurrent_gated_delta_rule(
                        inputs["query"],
                        inputs["key"],
                        inputs["value"],
                        inputs["state"],
                        beta=inputs["beta"],
                        scale=0.125,
                        actual_seq_lengths=inputs["actual_seq_lengths"],
                        ssm_state_indices=inputs["ssm_state_indices"],
                        num_accepted_tokens=inputs["num_accepted_tokens"],
                        g=inputs["g"],
                        gk=inputs["gk"],
                    )

        operator_argtypes = ACLNN_CTYPES._GET_WORKSPACE_ARGTYPES[
            "aclnnRecurrentGatedDeltaRule"
        ][:-2]
        self.assertEqual(captured["name"], "aclnnRecurrentGatedDeltaRule")
        self.assertEqual(output.shape, inputs["value"].shape)
        self.assertEqual(len(captured["args"]), len(operator_argtypes))
        self.assertEqual([type(arg) for arg in captured["args"]], operator_argtypes)
        self.assertEqual(
            captured["descriptor_names"],
            [
                "query",
                "key",
                "value",
                "beta",
                "state",
                "actual_seq_lengths",
                "ssm_state_indices",
                "g",
                "gk",
                "num_accepted_tokens",
                "out",
            ],
        )
        for name, tensor, format_override, storage_shape in captured[
            "descriptor_metadata"
        ]:
            self.assertEqual(format_override, ACLNN_CTYPES.ACL_FORMAT_ND, name)
            expected_storage_shape = None if name == "state" else tensor.shape
            self.assertEqual(storage_shape, expected_storage_shape, name)

    def test_causal_conv1d_bwd_signature_matches_aclnn_prototype(self):
        expected_argtypes = [
            *([ctypes.c_void_p] * 7),
            ctypes.c_int64,
            ctypes.c_char_p,
            *([ctypes.c_void_p] * 4),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.assertEqual(
            ACLNN_CTYPES._GET_WORKSPACE_ARGTYPES["aclnnCausalConv1dBwd"],
            expected_argtypes,
        )

    def test_causal_conv1d_bwd_wrapper_builds_one_value_per_operator_argtype(self):
        captured = {}

        def fake_empty(shape, like, **kwargs):
            del like, kwargs
            return FakeTensor(shape)

        def fake_call_aclnn(name, build_args, outputs):
            context = FakeCallContext()
            captured["name"] = name
            captured["args"] = build_args(context)
            captured["descriptor_names"] = context.descriptor_names
            return outputs

        x = FakeTensor((2, 17, 80))
        weight = FakeTensor((4, 80))
        dy = FakeTensor((2, 17, 80))
        with mock.patch.object(ACLNN_CTYPES, "_empty", side_effect=fake_empty):
            with mock.patch.object(ACLNN_CTYPES, "_call_aclnn", side_effect=fake_call_aclnn):
                outputs = ACLNN_CTYPES.npu_causal_conv1d_bwd(
                    x=x,
                    y=None,
                    weight=weight,
                    dy=dy,
                    input_layout="BSH",
                )

        operator_argtypes = ACLNN_CTYPES._GET_WORKSPACE_ARGTYPES["aclnnCausalConv1dBwd"][:-2]
        self.assertEqual(captured["name"], "aclnnCausalConv1dBwd")
        self.assertEqual(len(outputs), 4)
        self.assertEqual(len(captured["args"]), len(operator_argtypes))
        self.assertEqual([type(arg) for arg in captured["args"]], operator_argtypes)
        self.assertEqual(
            captured["descriptor_names"],
            [
                "x",
                "y",
                "weight",
                "dy",
                "initial_state",
                "dht",
                "query_start_loc",
                "dx",
                "dw",
                "db",
                "dh0",
            ],
        )


if __name__ == "__main__":
    unittest.main()
