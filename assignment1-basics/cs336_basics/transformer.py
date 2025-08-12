import torch
import torch.nn as nn
from .nn_utils import Linear, Embedding, RMSNorm, SwiGLU, RotaryPositionalEmbedding
from .nn_utils import softmax, scaled_dot_product_attention

class MultiHeadSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention module.
    """
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        """
        Initialize a Multi-Head Self-Attention module.
        
        Args:
            d_model: Dimensionality of the model
            num_heads: Number of attention heads
            device: Device to place the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        
        # Ensure d_model is divisible by num_heads for even splitting
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        # Store dimensions
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Linear projections for queries, keys, values
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        
    def forward(self, x: torch.Tensor, use_rope: bool = False, rope: RotaryPositionalEmbedding = None, token_positions=None) -> torch.Tensor:
        """
        Compute causal multi-head self-attention.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            use_rope: Whether to use RoPE (rotary positional embeddings)
            rope: RoPE module to use if use_rope is True
            token_positions: Token positions for RoPE if use_rope is True
            
        Returns:
            Output tensor of shape (..., seq_len, d_model)
        """
        batch_size = x.size(0)
        seq_len = x.size(-2)
        
        # Project queries, keys, and values
        q = self.q_proj(x)  # (..., seq_len, d_model)
        k = self.k_proj(x)  # (..., seq_len, d_model)
        v = self.v_proj(x)  # (..., seq_len, d_model)
        
        # Reshape to split heads
        # (..., seq_len, num_heads, head_dim)
        q = q.view(*x.shape[:-1], self.num_heads, self.head_dim)
        k = k.view(*x.shape[:-1], self.num_heads, self.head_dim)
        v = v.view(*x.shape[:-1], self.num_heads, self.head_dim)
        
        # Reshape to move head dim after batch dim (for easy parallel computation)
        # (..., num_heads, seq_len, head_dim)
        q = q.transpose(-3, -2)
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)
        
        # Apply RoPE to queries and keys if specified
        if use_rope and rope is not None:
            # Apply RoPE to each head separately
            q = rope(q, token_positions)
            k = rope(k, token_positions)
            
        # Create causal mask (lower triangular)
        # shape: (seq_len, seq_len)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        causal_mask = ~causal_mask  # True means attend, False means mask out
        
        # Expand mask for broadcasting with attention scores
        # shape: (1, 1, seq_len, seq_len) - will broadcast to batch and head dimensions
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply scaled dot-product attention with mask
        # attn_output shape: (..., num_heads, seq_len, head_dim)
        attn_output = scaled_dot_product_attention(q, k, v, causal_mask)
        
        # Reshape back to original shape
        # First, transpose back: (..., seq_len, num_heads, head_dim)
        attn_output = attn_output.transpose(-3, -2)
        
        # Combine heads: (..., seq_len, d_model)
        attn_output = attn_output.reshape(*x.shape[:-1], self.d_model)
        
        # Final projection
        output = self.output_proj(attn_output)
        
        return output