from typing import Any
import torch
import triton.language as tl
import triton
import math


# -------------------------------
# Triton Kernel: FlashAttention Forward Pass
# -------------------------------
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_q, N_k, scale,
    D: tl.constexpr, Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):

    # Tile & batch index
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb + query_tile_index * Q_TILE_SIZE * stride_qq,
        shape=(N_q, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(D, N_k), # transpose
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE), # transpose
        order=(0, 1)
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_k, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    # load Q_i
    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1))

    # initialize O_i, l_i, m_i
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE, ), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE, ), -float('inf'), dtype=tl.float32)

    # inter loop
    T_k = tl.cdiv(N_k, K_TILE_SIZE)
    for j in range(T_k):
        # load K_j, V_j
        K_j = tl.load(K_block_ptr, boundary_check=(1, 0))
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1))

        # compute tile of softmax scores S_ij
        S_ij = tl.dot(Q_i.to(tl.float32), K_j.to(tl.float32)) * scale

        # Causal masking
        if is_causal:
            q_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_indices = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_indices[:, None] >= k_indices[None, :]
            S_ij = tl.where(causal_mask, S_ij, -float('inf'))

        # compute m_i
        m_i_new = tl.maximum(m_i, tl.max(S_ij, -1))

        # compute P_ij
        P_ij = tl.exp(S_ij - m_i_new[:, None])
        
        exp_m = tl.exp(m_i - m_i_new)
        # compute l_i, O_i
        l_i = exp_m * l_i + tl.sum(P_ij, -1)
        O_i = O_i * exp_m[:, None] + tl.dot(P_ij.to(V_j.dtype), V_j)

        # update m_i
        m_i = m_i_new

        # advance pointer of K_j, V_j
        K_block_ptr = K_block_ptr.advance((0, K_TILE_SIZE))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    
    # normalize
    O_i = O_i / l_i[:, None]

    L_i = m_i + tl.log(l_i)

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob + query_tile_index * Q_TILE_SIZE * stride_oq,
        shape=(N_q, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb + query_tile_index * Q_TILE_SIZE * stride_lq, 
        shape=(N_q, ),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE, ),
        order=(0,)
    )

    # triton store
    tl.store(O_block_ptr, O_i.to(O_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, L_i, boundary_check=(0,))



# -------------------------------
# Autograd Function for FlashAttention
# -------------------------------

class FLashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, N_q, D = Q.shape
        _, N_k, _ = K.shape

        # initialize
        O = torch.empty_like(Q)
        L = torch.empty((B, N_q), device=Q.device, dtype=torch.float32)

        # tile size
        Q_TILE_SIZE, K_TILE_SIZE = 64, 64
        T_q = math.ceil(N_q / Q_TILE_SIZE)
        grid = (T_q, B)

        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_q, N_k, D ** -0.5,
            D=D, Q_TILE_SIZE=Q_TILE_SIZE, K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal

        return O
    
