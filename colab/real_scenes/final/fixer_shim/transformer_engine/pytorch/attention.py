import torch


class DotProductAttention(torch.nn.Module):
    def __init__(self, *a, **k): super().__init__(); raise RuntimeError('transformer_engine attention not available: use the torch backend')


def apply_rotary_pos_emb(t, freqs, tensor_format='sbhd', fused=False, **kw):
    """TE semantics (non-interleaved): freqs [S,1,1,D] angles; rotate first D dims of t as two halves."""
    if tensor_format == 'bshd': freqs = freqs.transpose(0, 1)                               # [S,1,1,D] -> [1,S,1,D]
    d = freqs.shape[-1]; tr, rest = t[..., :d], t[..., d:]
    cos, sin = freqs.cos().to(t.dtype), freqs.sin().to(t.dtype)
    x1, x2 = tr.chunk(2, -1); rot = torch.cat((-x2, x1), -1)
    return torch.cat((tr * cos + rot * sin, rest), -1)
