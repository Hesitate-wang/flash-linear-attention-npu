import os
import sys
import logging
import numpy as np
import random
import ml_dtypes
import torch
import torch_npu
from ml_dtypes import bfloat16
from dataclasses import dataclass
import math
import shutil
from typing import Optional
import fla_npu
np.random.seed(1)
torch.manual_seed(1)

WORKSPACE = os.path.dirname(os.path.abspath(__file__))

def cdiv_torch(a, b):
    return (a + b - 1) // b

def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in cdiv_torch(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)

def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]

def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int
) -> torch.LongTensor:
    return torch.cat([cu_seqlens.new_tensor([0]), cdiv_torch(prepare_lens(cu_seqlens), chunk_size)]).cumsum(-1) 

def forward_h_trans_cpu(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
):
    # 典型场景 HQ=HK=16 HV=32
    # assert HV >= HK 并且可以整除
    # assert 变长场景下，B==1
    # assert K==V==D
    dtype_ = k.dtype
    state_type_ = initial_state.dtype if initial_state is not None else torch.float32
    assert chunk_size in (32, 64, 128), f"unsupported chunk_size={chunk_size}"

    # FP64 reference, same precision as the ATK golden (_forward_h_ref).
    k = k.to(torch.float64)
    w = w.to(torch.float64)
    u = u.to(torch.float64)
    g = g.to(torch.float64)

    B, HK, T, K = k.shape[0], k.shape[1], k.shape[2], k.shape[3]
    HV, V = u.shape[1], u.shape[3]

    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, (T + BT - 1) // BT, None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    if initial_state is not None:
        # Canonical layout is [N, HV, K, V]. Accept the legacy
        # [shape_batch, HV, token_batch, K, V] and pick the sequence axis explicitly:
        # dense keeps shape_batch, varlen keeps token_batch. Reshaping the legacy
        # layout directly swaps HV and N whenever both are > 1.
        if initial_state.dim() == 5:
            sb, hv_legacy, tb = initial_state.shape[0], initial_state.shape[1], initial_state.shape[2]
            if tb == 1:
                initial_state = initial_state.reshape(sb, hv_legacy, K, V)
            elif sb == 1:
                initial_state = initial_state.reshape(hv_legacy, tb, K, V).transpose(0, 1)
            else:
                raise ValueError(
                    f"ambiguous initial_state layout {tuple(initial_state.shape)}: "
                    "expected [N,HV,K,V] or legacy [B,HV,1,K,V] / [1,HV,N,K,V]")
        initial_state = initial_state.reshape([N, HV, K, V]).contiguous().to(torch.float64)

    S = torch.zeros((B, HV, NT, K, V), device=k.device, dtype=torch.float64)
    v_new_output = torch.zeros((B, HV, T, V), device=k.device, dtype=torch.float64)
    final_state = torch.zeros((N, HV, K, V), device=k.device, dtype=torch.float64)

    head_ratio = HV // HK
    for n in range(N):
        if cu_seqlens is None: # 定长
            bos = 0
            eos = T
            T_inner = T
            NT_inner = NT
            boh = 0
        else:
            bos = cu_seqlens[n]
            eos = cu_seqlens[n + 1]
            T_inner = eos - bos
            NT_inner = (T_inner + BT - 1) // BT
            boh = chunk_offsets[n]

        if NT_inner == 0:
            # 空序列没有 chunk：内核只对 fp32 final_state 走"initial_state 原样回写"
            # 的路径（AIV 前导的直接 UB 分支），其它 dtype 保持 0。
            if initial_state is not None and state_type_ == torch.float32:
                final_state[n] = initial_state[n]
            continue

        for h in range(HV):
            # B H T D ->
            for i in range(NT_inner):
                k_sel = torch.zeros((BT, k.shape[-1]), device=k.device, dtype=k.dtype)
                w_sel = torch.zeros((BT, w.shape[-1]), device=w.device, dtype=w.dtype)
                u_sel = torch.zeros((BT, u.shape[-1]), device=u.device, dtype=u.dtype)
                g_sel = torch.zeros((BT), device=g.device, dtype=g.dtype)
                actual_len = min(bos + (i + 1) * BT, eos) - (bos + i * BT)

                if cu_seqlens is None: # 定长
                    k_sel[:actual_len, :] = k[n, h // head_ratio, bos + i * BT : bos + i * BT + actual_len, :]
                    w_sel[:actual_len, :] = w[n, h, bos + i * BT : bos + i * BT + actual_len, :]
                    u_sel[:actual_len, :] = u[n, h, bos + i * BT : bos + i * BT + actual_len, :]
                    g_sel[:actual_len] = g[n, h, bos + i * BT : bos + i * BT + actual_len]
                    if initial_state is not None and i == 0:
                        S[n, h, boh+i] = initial_state[n, h]
                    v_new = u_sel - w_sel @ S[n, h, boh+i]
                    if i != NT_inner-1:
                        S[n, h, boh+i+1] = S[n, h, boh+i] * g_sel[actual_len-1, None, None].exp() + k_sel.transpose(-1, -2) @ (v_new * (g_sel[actual_len-1, None] - g_sel).exp()[..., None])
                    else:
                        final_state[n, h] = S[n, h, boh+i] * g_sel[actual_len-1, None, None].exp() + k_sel.transpose(-1, -2) @ (v_new * (g_sel[actual_len-1, None] - g_sel).exp()[..., None])
                    v_new_output[n, h, bos + i * BT: bos + i * BT + actual_len, :] = v_new[:actual_len, :]

                else:
                    k_sel[:actual_len, :] = k[0, h // head_ratio, bos + i * BT : bos + i * BT + actual_len, :]
                    w_sel[:actual_len, :] = w[0, h, bos + i * BT : bos + i * BT + actual_len, :]
                    u_sel[:actual_len, :] = u[0, h, bos + i * BT : bos + i * BT + actual_len, :]
                    g_sel[:actual_len] = g[0, h, bos + i * BT : bos + i * BT + actual_len]
                    if initial_state is not None and i == 0:
                        S[0, h, boh+i] = initial_state[n, h]
                    v_new = u_sel - w_sel @ S[0, h, boh+i]
                    if i != NT_inner-1:
                        S[0, h, boh+i+1] = S[0, h, boh+i] * g_sel[actual_len-1, None, None].exp() + k_sel.transpose(-1, -2) @ (v_new * (g_sel[actual_len-1, None] - g_sel).exp()[..., None])
                    else:
                        final_state[n, h] = S[0, h, boh+i] * g_sel[actual_len-1, None, None].exp() + k_sel.transpose(-1, -2) @ (v_new * (g_sel[actual_len-1, None] - g_sel).exp()[..., None])
                    v_new_output[0, h ,bos + i * BT: bos + i * BT + actual_len, :] = v_new[:actual_len, :]


    #S = S.to(torch.bfloat16)
    #v_new_output = v_new_output.to(torch.bfloat16)
    S = S.transpose(1, 2).contiguous().to(dtype_)
    v_new_output = v_new_output.to(dtype_)
    final_state = final_state.to(state_type_) if final_state is not None else None
    return S, v_new_output, final_state

def parse_dtype(str_dtype):
    if str_dtype == "half" or str_dtype == "fp16" or str_dtype == "float16":
        return torch.float16
    elif str_dtype == "bf16" or str_dtype == "bfloat16":
        return torch.bfloat16
    if str_dtype == "float" or str_dtype == "float32":
        return torch.float32
    else:
        logging("[ERROR] dtype must be half or bf16")
        sys.exit()

class GDNFwdHInput:
    def __init__(self):
        self.batch = int(sys.argv[1])
        self.seqlen = int(sys.argv[2])
        self.k_num_head = int(sys.argv[3])
        self.v_num_head = int(sys.argv[4])
        self.k_head_dim = int(sys.argv[5])
        self.v_head_dim = int(sys.argv[6])
        self.is_varied_len = int(sys.argv[7])
        self.chunk_size = int(sys.argv[8])
        self.use_initial_state = bool(int(sys.argv[9]))
        self.store_final_state = bool(int(sys.argv[10]))
        self.dtype = parse_dtype(str(sys.argv[11]))
        self.use_actual_input = int(sys.argv[12])
        self.use_actual_output = int(sys.argv[13])
        self.data_path = str(sys.argv[14])
        torch.npu.set_device(int(sys.argv[15]))
        self.g_dtype = parse_dtype(str(sys.argv[16]))
        self.state_dtype = parse_dtype(str(sys.argv[17]))

        if self.is_varied_len:
            self.shape_batch = 1
            self.token_batch = self.batch
        else:
            self.shape_batch = self.batch
            self.token_batch = 1

class GDNFwdHInputTensor:
    def __init__(self, k, w, u, g, cu_seqlens, chunk_offsets, initial_state):
        self.k = k
        self.w = w
        self.u = u
        self.g = g
        self.cu_seqlens = cu_seqlens
        self.chunk_offsets = chunk_offsets
        self.initial_state = initial_state

class GDNFwdHOutputTensor:
    def __init__(self, h, v_new, final_state = None):
        self.h = h
        self.v_new = v_new
        self.final_state = final_state

def parse_actual_input(h_input):
    actual_data = torch.load(h_input.data_path, map_location='cpu')
    k = actual_data['k'][:, :, :h_input.k_num_head].to(h_input.dtype).transpose(1, 2).contiguous()
    w = actual_data['w'][:, :, :h_input.v_num_head].to(h_input.dtype).transpose(1, 2).contiguous()
    u = actual_data['u'][:, :, :h_input.v_num_head].to(h_input.dtype).transpose(1, 2).contiguous()
    g = actual_data['g'][:, :, :h_input.v_num_head].to(h_input.g_dtype).transpose(1, 2).contiguous()
    cu_seqlens, chunk_offsets = get_cu_offsets(h_input, actual_data.get('cu_seqlens'))
    initial_state = None
    if h_input.use_initial_state:
        if actual_data.get('initial_state', None) is not None:
            initial_state = actual_data['initial_state'].reshape([h_input.shape_batch, h_input.v_num_head, h_input.token_batch, h_input.k_head_dim, h_input.v_head_dim]).contiguous().to(h_input.state_dtype)
        else:
            initial_state = torch.randn([h_input.shape_batch, h_input.v_num_head, h_input.token_batch, h_input.k_head_dim, h_input.v_head_dim], dtype=h_input.state_dtype)
    
    return GDNFwdHInputTensor(k, w, u, g, cu_seqlens, chunk_offsets, initial_state)

def parse_actual_output(h_input):
    actual_data = torch.load(h_input.data_path, map_location='cpu')
    h = actual_data['h'] if 'h' in actual_data.keys() else actual_data['ref_h']
    v = actual_data['v_new'] if 'v_new' in actual_data.keys() else actual_data['ref_v_new']
    final_state = actual_data.get('final_state', None)
    if final_state is not None:
        final_state = final_state.to(h_input.state_dtype)
    h = h[:, :, :h_input.v_num_head].to(h_input.dtype).contiguous()
    v = v[:, :, :h_input.v_num_head].to(h_input.dtype).transpose(1, 2).contiguous()
    return GDNFwdHOutputTensor(h, v, final_state)

def gen_seqlen(seqlen, is_varied_len, batch):
    if is_varied_len == 0:
        return None
    cu_seqlens = [0]
    avg_len = seqlen // batch
    for i in range(batch - 1):
        diff = random.randint(avg_len // 2, avg_len * 3 // 2)
        cu_seqlens.append(cu_seqlens[-1] + diff)
    cu_seqlens.append(seqlen)
    return torch.Tensor(cu_seqlens).to(torch.int64)

def gen_wu_data():
    pass

def get_cu_offsets(h_input, cu_seqlens):
    if cu_seqlens is None:
        return None, None
    cu_seqlens = cu_seqlens.to(torch.int64)
    num_chunks = 0
    curr_token = 0
    for seq in cu_seqlens:
        num_chunks += math.ceil((seq - curr_token) / h_input.chunk_size)
        curr_token = seq
    return cu_seqlens, torch.zeros([num_chunks, 2]).to(cu_seqlens.dtype)

def gen_decay_data(h_input, cu_seqlens, chunk_offsets):
    """沿序列单调递减的累计 gate，与 ATK `_gate` 同分布。

    每 token 的 log 衰减增量取 [0.001, 0.011]，按序列（而不是按 chunk）累加：
    - 逐 chunk 累加会让 g 在 chunk 边界锯齿跳变，且量级到 -350，fp32 下
      exp(g_last) 直接下溢为 0；
    - 增量过大会让 q = exp(g_last - g_t) 只覆盖最后一个 token，递推失去意义。
    """
    g = torch.zeros([h_input.shape_batch, h_input.v_num_head, h_input.seqlen],
                    dtype=torch.float32)
    if cu_seqlens is None:
        ranges = [(0, h_input.seqlen)]
    else:
        ranges = [(int(cu_seqlens[i]), int(cu_seqlens[i + 1]))
                  for i in range(len(cu_seqlens) - 1)]
    for shape_batch_idx in range(h_input.shape_batch):
        for v_head_idx in range(h_input.v_num_head):
            for start, end in ranges:
                length = end - start
                if length <= 0:
                    continue
                inc = torch.empty([length]).uniform_(0.001, 0.011)
                g[shape_batch_idx, v_head_idx, start:end] = -torch.cumsum(inc, 0)
    return g

def gen_input_data(h_input, rand_wu = True):
    cu_seqlens = gen_seqlen(h_input.seqlen, h_input.is_varied_len, h_input.token_batch)
    cu_seqlens, chunk_offsets = get_cu_offsets(h_input, cu_seqlens)
    # 与 ATK `_randn(scale=0.05)` 一致：标准正态会让 h <- D·h + Kᵀq(u - W·h)
    # 的随机放大因子达到 ~1+128σ²，20 个 chunk 后溢出到 1e17，参考和 DUT 都失去意义。
    input_scale = 0.05
    if rand_wu:
        w = torch.randn([h_input.shape_batch, h_input.v_num_head, h_input.seqlen, h_input.k_head_dim], dtype=h_input.dtype) * input_scale
        u = torch.randn([h_input.shape_batch, h_input.v_num_head, h_input.seqlen, h_input.v_head_dim], dtype=h_input.dtype) * input_scale
    else:
        w, u = gen_wu_data()
    k = torch.randn([h_input.shape_batch, h_input.k_num_head, h_input.seqlen, h_input.k_head_dim], dtype=h_input.dtype) * input_scale
    # g = torch.randn([h_input.shape_batch, h_input.v_num_head, h_input.seqlen], dtype=torch.float)
    g = gen_decay_data(h_input, cu_seqlens, chunk_offsets)
    if h_input.use_initial_state:
        # 直接生成 canonical [N, HV, K, V]：dense 时 N=shape_batch，变长时 N=token_batch。
        n_seq = h_input.token_batch if h_input.is_varied_len else h_input.shape_batch
        initial_state = torch.randn(
            [n_seq, h_input.v_num_head, h_input.k_head_dim, h_input.v_head_dim],
            dtype=h_input.state_dtype)
    else:
        initial_state = None
    return GDNFwdHInputTensor(k, w, u, g, cu_seqlens, chunk_offsets, initial_state)

def gen_ref_data(h_input, input_tensor):
    # chunk_size 必须显式传下去：forward_h_trans_cpu 的默认值 64 会让
    # chunk=32/128 的参考按 64 计算。
    h, v, final_state = forward_h_trans_cpu(
        k=input_tensor.k,
        w=input_tensor.w,
        u=input_tensor.u,
        g=input_tensor.g,
        initial_state=input_tensor.initial_state,
        output_final_state=h_input.store_final_state,
        chunk_size=h_input.chunk_size,
        cu_seqlens=input_tensor.cu_seqlens,
    )
    if not h_input.store_final_state:
        final_state = None
    return GDNFwdHOutputTensor(h, v, final_state)

def save_data(input_tensor, output_tensor):
    data_dir = os.path.join(WORKSPACE, "data")
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    input_tensor.k.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "k.bin"))
    input_tensor.w.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "w.bin"))
    input_tensor.u.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "u.bin"))
    input_tensor.g.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "g.bin"))
    if input_tensor.cu_seqlens is not None:
        np.array(input_tensor.cu_seqlens.cpu()).astype(np.int64).tofile(os.path.join(WORKSPACE, "data", "cu_seqlens.bin"))

    if input_tensor.initial_state is not None:
        input_tensor.initial_state.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "initial_state.bin"))

    output_tensor.h.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "h_ref.bin"))
    output_tensor.v_new.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "v_ref.bin"))
    if output_tensor.final_state is not None:
        output_tensor.final_state.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "final_state_ref.bin"))

if __name__ == "__main__":

    gdn_fwd_h_input = GDNFwdHInput()

    if gdn_fwd_h_input.use_actual_input:
        input_tensor = parse_actual_input(gdn_fwd_h_input)
    else:
        input_tensor = gen_input_data(gdn_fwd_h_input)

    def _as_int_list(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.cpu().tolist()
        return list(x)

    torch.npu.synchronize()
    # `torch.ops.npu.npu_*` 是 legacy 兼容路径（需要 fla_npu.load_legacy_torch_ops() 才注册，
    # 且调用会告警）。当前公开入口是 fla_npu.ops.ascendc；与 ATK executor 和
    # torch_custom/fla_npu/test/test_fwd_h.py 保持一致。
    from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd_h

    # chunk_indices 由 wrapper 依据 cu_seqlens + chunk_size 规范化生成；
    # get_cu_offsets() 返回的占位零张量不能当真实索引传。
    result = chunk_gated_delta_rule_fwd_h(
        input_tensor.k.npu(),
        input_tensor.w.npu(),
        input_tensor.u.npu(),
        g=input_tensor.g.npu(),
        initial_state=(
            input_tensor.initial_state.npu()
            if input_tensor.initial_state is not None
            else None
        ),
        output_final_state=gdn_fwd_h_input.store_final_state,
        chunk_size=gdn_fwd_h_input.chunk_size,
        cu_seqlens=_as_int_list(input_tensor.cu_seqlens),
    )
    torch.npu.synchronize()
    if gdn_fwd_h_input.use_actual_output:
        output_tensor = parse_actual_output(gdn_fwd_h_input)
    else:
        output_tensor = gen_ref_data(gdn_fwd_h_input, input_tensor)
    save_data(input_tensor, output_tensor)
    result[0].cpu().view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "h_npu.bin"))
    result[1].cpu().view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "v_npu.bin"))
    if result[2] is not None:
        result[2].cpu().view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "final_state_npu.bin"))
