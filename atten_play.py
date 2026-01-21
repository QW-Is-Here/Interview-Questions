import torch 
from torch.nn import functional as F
import torch.nn as nn


# 1. generate Q K V linear projection
# 2. Q x K^T
# 3. Q x K^T -> causal mask -> softmax -> dropout
# 4. softmax(Q x K^T / dk**0.5) x V 

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, attn_dim, dropout_rate):
        super().__init__()

        self.attn_dim = attn_dim

        self.q_proj = nn.Linear(hidden_dim, attn_dim)
        self.k_proj = nn.Linear(hidden_dim, attn_dim)
        self.v_proj = nn.Linear(hidden_dim, attn_dim)

        self.attn_scale = attn_dim**-0.5

        self.dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(attn_dim, hidden_dim)
    
    def forward(self, x, causal_mask = None):
        # x (bs, seq_len, hidden_dim)
        Q = self.q_proj(x)  # Q (bs, seq_len, attn_dim)
        K = self.k_proj(x)  # K (bs, seq_len, attn_dim)
        V = self.v_proj(x)  # V (bs, seq_len, attn_dim)

        # attn weight shape is (bs,seq_len, seq_len)
        attn_weight= torch.matmul(Q, K.transpose(-1, -2)) * self.attn_scale
        if causal_mask is not None:
            attn_weight += causal_mask
        attn_weight = torch.softmax(attn_weight, dim = -1)
        attn_weight = self.dropout(attn_weight)

        # shape: from (bs, seq_len, seq_len) --> (bs, seq_len attn_dim)
        output = torch.matmul(attn_weight, V)
        output = self.output_layer(output)
        return output
    

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, attn_dim, num_head, dropout_rate):
        super().__init__()
        self.attn_dim = attn_dim
        self.num_head = num_head

        self.q_proj = nn.Linear(hidden_dim, num_head * attn_dim)
        self.k_proj = nn.Linear(hidden_dim, num_head * attn_dim)
        self.v_proj = nn.Linear(hidden_dim, num_head * attn_dim)

        self.attn_scale = attn_dim**-0.5

        self.dropout_layer = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(num_head * attn_dim, hidden_dim)

    def forward(self, x, causal_mask=None):
        bs, seq_len, hidden_dim = x.shape
        Q = self.q_proj(x)  # Q (bs, seq_len, attn_dim * num_head)
        K = self.k_proj(x)  # K (bs, seq_len, attn_dim * num_head)
        V = self.v_proj(x)  # V (bs, seq_len, attn_dim * num_head)

        Q = Q.view(bs, seq_len, self.num_head, self.attn_dim).transpose(1,2)
        K = K.view(bs, seq_len, self.num_head, self.attn_dim).transpose(1,2)
        V = V.view(bs, seq_len, self.num_head, self.attn_dim).transpose(1,2)

        attn_weight = torch.matmul(Q, K.transpose(-1, -2)) * self.attn_scale

        if causal_mask is not None:
            # causal_mask = causal_mask.unsqueeze(0).repeat(bs,1,1)
            causal_mask = causal_mask.unsqueeze(1).repeat(1, self.num_head, 1, 1)
            attn_weight += causal_mask
        
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = self.dropout_layer(attn_weight)

        result = torch.matmul(attn_weight, V)
        result = result.view(bs, seq_len, -1)
        result = self.output_layer(result)
        return result
    

class MultiHeadAttentionWithKVCache(nn.Module):
    """
    Multi-Head Attention with KV Cache for efficient auto-regressive generation
    - During training: processes full sequences (like standard MHA)
    - During inference: caches past K,V to avoid recomputation
    """
    def __init__(self, hidden_dim, attn_dim, num_head, dropout_rate, max_seq_len=2048):
        super().__init__()
        self.attn_dim = attn_dim
        self.num_head = num_head
        self.max_seq_len = max_seq_len

        self.q_proj = nn.Linear(hidden_dim, num_head * attn_dim)
        self.k_proj = nn.Linear(hidden_dim, num_head * attn_dim)
        self.v_proj = nn.Linear(hidden_dim, num_head * attn_dim)

        self.attn_scale = attn_dim**-0.5

        self.dropout_layer = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(num_head * attn_dim, hidden_dim)

        # KV cache buffers (will be initialized during first use)
        self.register_buffer('k_cache', None, persistent=False)
        self.register_buffer('v_cache', None, persistent=False)
        self.register_buffer('cache_position', torch.tensor(0), persistent=False)

    def reset_cache(self):
        """Reset the KV cache (call this at the start of each new sequence)"""
        self.k_cache = None
        self.v_cache = None
        self.cache_position = torch.tensor(0)

    def forward(self, x, causal_mask=None, use_cache=False, cache_position=None):
        """
        Args:
            x: input tensor (bs, seq_len, hidden_dim)
            causal_mask: optional causal mask
            use_cache: whether to use KV cache (for inference)
            cache_position: current position in cache (for inference)
        
        Returns:
            output: attention output (bs, seq_len, hidden_dim)
        """
        bs, seq_len, hidden_dim = x.shape
        
        # Project Q, K, V
        Q = self.q_proj(x)  # (bs, seq_len, attn_dim * num_head)
        K = self.k_proj(x)  # (bs, seq_len, attn_dim * num_head)
        V = self.v_proj(x)  # (bs, seq_len, attn_dim * num_head)

        # Reshape to multi-head format
        Q = Q.view(bs, seq_len, self.num_head, self.attn_dim).transpose(1, 2)  # (bs, num_head, seq_len, attn_dim)
        K = K.view(bs, seq_len, self.num_head, self.attn_dim).transpose(1, 2)  # (bs, num_head, seq_len, attn_dim)
        V = V.view(bs, seq_len, self.num_head, self.attn_dim).transpose(1, 2)  # (bs, num_head, seq_len, attn_dim)

        if use_cache:
            # Inference mode with KV cache
            if self.k_cache is None or self.v_cache is None:
                # Initialize cache on first use
                self.k_cache = torch.zeros(
                    bs, self.num_head, self.max_seq_len, self.attn_dim,
                    dtype=K.dtype, device=K.device
                )
                self.v_cache = torch.zeros(
                    bs, self.num_head, self.max_seq_len, self.attn_dim,
                    dtype=V.dtype, device=V.device
                )
                self.cache_position = torch.tensor(0, device=K.device)

            # Get current position
            if cache_position is not None:
                pos = cache_position
            else:
                pos = self.cache_position.item()

            # Update cache with new K, V
            self.k_cache[:, :, pos:pos+seq_len, :] = K
            self.v_cache[:, :, pos:pos+seq_len, :] = V

            # Use cached K, V up to current position
            K = self.k_cache[:, :, :pos+seq_len, :]
            V = self.v_cache[:, :, :pos+seq_len, :]

            # Update cache position
            self.cache_position = torch.tensor(pos + seq_len, device=K.device)

            # For inference, we typically only query the last token
            # So Q shape is (bs, num_head, 1, attn_dim) attending to all previous K,V
            # The causal mask should allow attending to all previous positions

        # Compute attention weights
        attn_weight = torch.matmul(Q, K.transpose(-1, -2)) * self.attn_scale  # (bs, num_head, seq_len_q, seq_len_k)

        if causal_mask is not None:
            # Adjust causal mask for cache scenario
            if use_cache and causal_mask.shape[-1] != attn_weight.shape[-1]:
                # Create appropriate mask for cached KV
                seq_len_q = Q.shape[2]
                seq_len_k = K.shape[2]
                # For autoregressive, each query position can attend to all previous K positions
                if seq_len_q == 1:
                    # Single token generation - can attend to all previous tokens
                    # No masking needed in this case
                    pass
                else:
                    causal_mask = causal_mask.unsqueeze(1).repeat(1, self.num_head, 1, 1)
                    attn_weight = attn_weight + causal_mask
            else:
                causal_mask = causal_mask.unsqueeze(1).repeat(1, self.num_head, 1, 1)
                attn_weight = attn_weight + causal_mask
        
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = self.dropout_layer(attn_weight)

        # Compute attention output
        result = torch.matmul(attn_weight, V)  # (bs, num_head, seq_len, attn_dim)
        result = result.transpose(1, 2).contiguous().view(bs, -1, self.num_head * self.attn_dim)
        result = self.output_layer(result)
        
        return result


class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim, attn_dim, num_q_a_group, num_head, droupout_rate):
        super().__init__()
        self.attn_dim = attn_dim
        self.num_head = num_head
        self.num_kv_head = num_head // num_q_a_group
        self.num_q_a_group = num_q_a_group

        self.attn_scale = attn_dim**-0.5

        self.q_proj = nn.Linear(hidden_dim, num_head * attn_dim)
        self.k_proj = nn.Linear(hidden_dim, self.num_kv_head * attn_dim)
        self.v_proj = nn.Linear(hidden_dim, self.num_kv_head * attn_dim)

        self.droupout_layer = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(attn_dim * num_head, hidden_dim)

    
    def forward(self, x, causal_mask=None):
        bs, seq_len, hidden_dim = x.shape
        Q = self.q_proj(x)  # Q (bs, seq_len, attn_dim * num_head)
        K = self.k_proj(x)  # K (bs, seq_len, attn_dim * num_kv_head)
        V = self.v_proj(x)  # V (bs, seq_len, attn_dim * num_kv_head)

        Q = Q.view(bs, seq_len, self.num_head, self.attn_dim).transpose(1,2)  # (bs, num_head, seq_len, attn_dim)
        K = K.view(bs, seq_len, self.num_kv_head, self.attn_dim).transpose(1,2)  # (bs, num_kv_head, seq_len, attn_dim)
        V = V.view(bs, seq_len, self.num_kv_head, self.attn_dim).transpose(1,2)  # (bs, num_kv_head, seq_len, attn_dim)   

        K = K.repeat_interleace(self.num_q_a_group, dim=1)
        V = V.repeat_interleace(self.num_q_a_group, dim=1)

        attn_weight = torch.matmul(Q, K.transpose(-1, -2)) * self.attn_scale    
        if causal_mask is not None:
            # causal_mask = causal_mask.unsqueeze(0).repeat(bs,1,1)
            causal_mask = causal_mask.unsqueeze(1).repeat(1, self.num_head, 1, 1)
            attn_weight += causal_mask

        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = self.dropout_layer(attn_weight)

        result = torch.matmul(attn_weight, V) # (bs ,num_head, seq_len, attn_dim)
        result = result.transpose(1,2).view(bs, seq_len, -1)
        result = self.output_layer(result)


class LinearAttention(nn.Module):
    """
    Linear Attention: O(N) complexity instead of O(N^2)
    Key idea: Change computation order from softmax(QK^T)V to Q(K^TV)
    Uses feature map φ(x) instead of softmax for Q and K
    """
    def __init__(self, hidden_dim, attn_dim, num_head, dropout_rate, feature_map='elu'):
        super().__init__()
        self.attn_dim = attn_dim
        self.num_head = num_head
        self.feature_map_type = feature_map

        self.q_proj = nn.Linear(hidden_dim, num_head * attn_dim)
        self.k_proj = nn.Linear(hidden_dim, num_head * attn_dim)
        self.v_proj = nn.Linear(hidden_dim, num_head * attn_dim)

        self.dropout_layer = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(num_head * attn_dim, hidden_dim)

        # Small epsilon for numerical stability
        self.eps = 1e-6

    def feature_map(self, x):
        """
        Apply feature map to Q and K
        Common choices: elu+1, relu, or identity
        """
        if self.feature_map_type == 'elu':
            return F.elu(x) + 1
        elif self.feature_map_type == 'relu':
            return F.relu(x)
        else:
            # Identity: just use raw values (less stable)
            return x

    def forward(self, x, causal_mask=None):
        bs, seq_len, hidden_dim = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # (bs, seq_len, num_head * attn_dim)
        K = self.k_proj(x)  # (bs, seq_len, num_head * attn_dim)
        V = self.v_proj(x)  # (bs, seq_len, num_head * attn_dim)

        # Reshape to multi-head format
        Q = Q.view(bs, seq_len, self.num_head, self.attn_dim).transpose(1, 2)  # (bs, num_head, seq_len, attn_dim)
        K = K.view(bs, seq_len, self.num_head, self.attn_dim).transpose(1, 2)  # (bs, num_head, seq_len, attn_dim)
        V = V.view(bs, seq_len, self.num_head, self.attn_dim).transpose(1, 2)  # (bs, num_head, seq_len, attn_dim)

        # Apply feature map to make Q and K positive
        Q = self.feature_map(Q)
        K = self.feature_map(K)

        if causal_mask is not None:
            # For causal linear attention, compute cumulatively
            # Initialize cumulative sums
            KV = torch.zeros(bs, self.num_head, self.attn_dim, self.attn_dim, device=x.device, dtype=x.dtype)
            K_sum = torch.zeros(bs, self.num_head, self.attn_dim, device=x.device, dtype=x.dtype)
            
            output = []
            for i in range(seq_len):
                # Update cumulative K^T V
                KV = KV + torch.einsum('bhd,bhe->bhde', K[:, :, i], V[:, :, i])
                K_sum = K_sum + K[:, :, i]
                
                # Compute output for position i: Q_i (K^T V) / (Q_i K^T)
                numerator = torch.einsum('bhd,bhde->bhe', Q[:, :, i], KV)
                denominator = torch.einsum('bhd,bhd->bh', Q[:, :, i], K_sum)
                
                output_i = numerator / (denominator.unsqueeze(-1) + self.eps)
                output.append(output_i.unsqueeze(2))
            
            result = torch.cat(output, dim=2)  # (bs, num_head, seq_len, attn_dim)
        else:
            # Non-causal linear attention: compute in one go
            # Compute K^T V: (bs, num_head, attn_dim, attn_dim)
            KV = torch.einsum('bhnd,bhne->bhde', K, V)
            
            # Compute normalization: sum over sequence for each head
            # K_sum: (bs, num_head, attn_dim)
            K_sum = K.sum(dim=2)
            
            # Compute Q (K^T V)
            numerator = torch.einsum('bhnd,bhde->bhne', Q, KV)
            
            # Compute Q K^T for normalization
            denominator = torch.einsum('bhnd,bhd->bhn', Q, K_sum)
            
            # Normalize
            result = numerator / (denominator.unsqueeze(-1) + self.eps)

        # Apply dropout
        result = self.dropout_layer(result)
        
        # Reshape and project back
        result = result.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        result = self.output_layer(result)
        
        return result
    

if __name__ == '__main__':
    bs = 3
    seq_len = 15
    hidden_dim = 1024
    attn_dim = 512
    dropout_rate = 0.3
    num_head = 6
    num_q_a_group = 2

    x = torch.randn(bs, seq_len, hidden_dim)

    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
    causal_mask = causal_mask.unsqueeze(0).repeat(bs,1,1)

    self_attn = SelfAttention(hidden_dim, attn_dim, dropout_rate)
    output = self_attn(x, causal_mask)
    print('self attn shape: ', output.shape)

    multi_attn = MultiHeadAttention(hidden_dim, attn_dim, num_head, dropout_rate)
    output = multi_attn(x, causal_mask)
    print('multi attn shape: ', output.shape)

    # Test KV cache version
    print('\n--- Testing KV Cache ---')
    mha_kv_cache = MultiHeadAttentionWithKVCache(hidden_dim, attn_dim, num_head, dropout_rate, max_seq_len=50)
    
    # Training mode: process full sequence (no cache)
    output_train = mha_kv_cache(x, causal_mask, use_cache=False)
    print('multi attn with kv cache (training mode) shape: ', output_train.shape)
    
    # Inference mode: simulate auto-regressive generation with cache
    mha_kv_cache.reset_cache()
    
    # Step 1: Process initial prompt (e.g., first 5 tokens)
    prompt_len = 5
    x_prompt = x[:, :prompt_len, :]
    output_prompt = mha_kv_cache(x_prompt, use_cache=True, cache_position=0)
    print(f'inference step 1 (prompt {prompt_len} tokens) output shape: ', output_prompt.shape)
    
    # Step 2: Generate next token (single token at a time)
    for i in range(3):  # Generate 3 more tokens
        x_next = x[:, prompt_len+i:prompt_len+i+1, :]  # Single token
        output_next = mha_kv_cache(x_next, use_cache=True)
        print(f'inference step {i+2} (token {prompt_len+i+1}) output shape: ', output_next.shape)

    gqa= GroupQueryAttention(hidden_dim, attn_dim, num_q_a_group, num_head, dropout_rate)
    output = multi_attn(x, causal_mask)
    print('group attn shape: ', output.shape)

    linear_attn = LinearAttention(hidden_dim, attn_dim, num_head, dropout_rate, feature_map='elu')
    output = linear_attn(x, causal_mask)
    print('linear attn shape: ', output.shape)

