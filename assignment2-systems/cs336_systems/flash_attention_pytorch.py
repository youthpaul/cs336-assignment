import torch
import math



class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, N_q, d = Q.shape
        _, N_k, _ = K.shape
        device = Q.device

        B_q, B_k = 64, 64 # tile size
        T_q, T_k = math.ceil(N_q / B_q), math.ceil(N_k / B_k) # number of tiles

        O = torch.empty((B, N_q, d), device=device, dtype=torch.float32) # output tensor
        L = torch.empty((B, N_q), device=device, dtype=torch.float32) # attention scores

        # for each batch
        for batch in range(B):
            Q_b, K_b, V_b = Q[batch], K[batch], V[batch]
            # flash-attention outer loop
            for i in range(T_q):
                q_start, q_end = i * B_q, min((i + 1) * B_q, N_q) # [q_start, q_end)
                curr_B_q = q_end - q_start

                # load Q_i
                Q_i = Q_b[q_start: q_end, :]

                # initialize O_i, l_i, m_i
                O_i = torch.zeros((curr_B_q, d), device=device, dtype=torch.float32)
                l_i = torch.zeros(curr_B_q, device=device, dtype=torch.float32)
                m_i = torch.full((curr_B_q, ), -float('inf'), device=device, dtype=torch.float32)

                # flash-attention inter loop
                for j in range(T_k):
                    k_start, k_end = j * B_k, min((j + 1) * B_k, N_k)

                    # load K_j, V_j
                    K_j = K_b[k_start: k_end, :]
                    V_j = V_b[k_start: k_end, :]

                    # compute tile of pre-softmax attention scores
                    S_ij = (Q_i @ K_j.transpose(-2, -1)) * d**-0.5 # (B_q, B_k)

                    if is_causal:
                        # attention mask
                        q_indices = torch.arange(q_start, q_end, device=device)
                        k_indices = torch.arange(k_start, k_end, device=device)
                        causal_mask = q_indices[:, None] >= k_indices[None, :]
                        S_ij = torch.where(causal_mask, S_ij, -float('inf'))
                    
                    # compute m_ij
                    m_i_new = torch.maximum(m_i, S_ij.max(dim=-1).values)

                    # compute P_ij
                    P_ij = torch.exp(S_ij - m_i_new[:, None])

                    exp_m = torch.exp(m_i - m_i_new)
                    # compute l_i
                    l_i = exp_m * l_i + P_ij.sum(dim=-1)

                    # compute O_ij
                    O_i = torch.diag(exp_m) @ O_i + P_ij @ V_j

                    # update m_i
                    m_i = m_i_new

                O_i = torch.diag(1.0 / l_i) @ O_i # divide by softmax-sum

                # wrie O_i and L_i
                O[batch, q_start: q_end, :] = O_i.to(Q.dtype)
                L[batch, q_start: q_end] = m_i + torch.log(l_i)

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal

        return O

    
    @staticmethod
    def backward(ctx, **args):

        raise NotImplemented