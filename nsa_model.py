"""
nsa_model.py — Native Sparse Attention (NSA) + a tiny decoder-only transformer.

Open-box LLM experiment, RUNG 1.
Reference: "Native Sparse Attention: Hardware-Aligned and Natively Trainable
Sparse Attention", DeepSeek/PKU/UW, arXiv:2502.11089 (ACL 2025).

The ONLY question this rung answers:
    Does backward flow correctly through the NSA gate and all three branches?

So this file optimizes for *correctness and readability*, NOT speed. It is pure
PyTorch (torch + numpy only): no Triton, no custom CUDA, no fused kernels, no
memory-efficient tricks. Tiny matmuls at this scale are fine. Wherever NSA's real
implementation would gather/pack blocks for efficiency, we instead compute full
score matrices and mask — same math, far easier to read and verify.

------------------------------------------------------------------------------
NSA in one paragraph
------------------------------------------------------------------------------
For every query, attention is computed three ways and blended by a learned,
per-head gate:

  1. COMPRESSION  — squash consecutive blocks of key/value tokens into a few
                    coarse "summary" tokens (learnable pooling). Cheap global view.
  2. SELECTION    — pick the top-k *fine-grained* token blocks and attend to them
                    at full resolution. The block-importance scores are read off
                    the COMPRESSION attention (see the trap below).
  3. SLIDING WIN  — attend to the most recent w tokens. Local detail.

  out = g_cmp * o_cmp + g_slc * o_slc + g_win * o_win          (per head, g in [0,1])

NSA assumes GQA (grouped-query attention): many query heads share each key/value
head. Selection is decided *per group* (query heads in a group select the same
blocks), matching the paper.

------------------------------------------------------------------------------
THE TWO TRAPS (read these before touching the selection branch)
------------------------------------------------------------------------------
TRAP 1 — the top-k pick is non-differentiable.
    `torch.topk` gives us indices. Backpropagating "which index was chosen" is
    meaningless. So we DETACH the importance scores before topk; the chosen block
    indices carry NO gradient. Gradient still reaches the selection branch two
    honest ways:
      (a) via the gate — o_slc is blended in, so the gate learns how much to trust
          selection, and selection's k/v projections get gradient from the actual
          attention over the selected tokens.
      (b) via the compression branch — the block-importance scores ARE the
          compression attention probabilities, which are fully differentiable and
          trained through the compression branch. So the model learns to *rank*
          blocks well through compression, and selection rides on that ranking.
    We NEVER backprop through the hard index pick.

TRAP 2 — you must be able to SEE the gradient.
    After loss.backward() the train loop asserts every branch's params
    (compression, selection, window, gate, shared q-proj) have non-None, non-zero
    grads, and prints a per-branch grad-norm table each step. See param_groups().
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def masked_softmax(scores, keep_mask):
    """Softmax over the last dim, keeping only positions where keep_mask is True.

    Rows with NO valid positions (e.g. the very first query, before any block is
    complete) are returned as all-zeros instead of NaNs — so the corresponding
    attention output is a clean zero vector and contributes nothing.
    """
    neg = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(~keep_mask, neg)
    attn = torch.softmax(scores, dim=-1)
    # zero out fully-masked rows (softmax of all -inf would be garbage/uniform)
    row_has_valid = keep_mask.any(dim=-1, keepdim=True)
    return attn * row_has_valid


def expand_kv(x, group_size):
    """GQA expansion: [B, H_kv, S, d] -> [B, H_kv*group_size, S, d].

    repeat_interleave keeps groups contiguous, so query head h uses kv head
    h // group_size. That contiguity is relied on when we aggregate importance
    scores per group (view Hq as (Hkv, group_size)).
    """
    B, Hkv, S, d = x.shape
    return x.repeat_interleave(group_size, dim=1)


# ---------------------------------------------------------------------------
# compression branch: learnable block pooling
# ---------------------------------------------------------------------------

class BlockCompressor(nn.Module):
    """Squash each block of `block_size` tokens into ONE summary token.

    Learnable, and the trainable params here are COMPRESSION-branch params:
      - `intra_pos`  : an intra-block positional embedding (so the pooler can tell
                       position-within-block apart, per the NSA paper).
      - `pool_logits`: learned softmax weights over the block positions (a learned
                       weighted mean — generalizes plain mean pooling).
      - `proj`       : a linear mix after pooling.
    """

    def __init__(self, block_size, d_head):
        super().__init__()
        self.block_size = block_size
        self.intra_pos = nn.Parameter(torch.zeros(block_size, d_head))
        self.pool_logits = nn.Parameter(torch.zeros(block_size))  # -> mean at init
        self.proj = nn.Linear(d_head, d_head, bias=False)

    def forward(self, x_blocks):
        # x_blocks: [B, H_kv, n_blk, block_size, d_head]
        x = x_blocks + self.intra_pos                    # add intra-block position
        w = torch.softmax(self.pool_logits, dim=0)       # [block_size], sums to 1
        pooled = (x * w[:, None]).sum(dim=-2)            # [B, H_kv, n_blk, d_head]
        return self.proj(pooled)


# ---------------------------------------------------------------------------
# the NSA attention module
# ---------------------------------------------------------------------------

class NSAAttention(nn.Module):
    """One NSA attention layer.

    Design choices for a *correct, readable* rung-1 prototype:
      * Compression and selection share the SAME block grid: non-overlapping
        blocks of size `block_size`, stride == block_size. NSA's general form
        allows overlapping compression blocks (stride d < block l) and then needs
        a formula to map compression scores onto selection blocks. By aligning the
        grids we get that mapping for free: importance(block j) == compression
        attention prob on summary token j. Documented here so the simplification
        is explicit.
      * Each branch has its OWN k/v projections (so each branch is a nameable
        parameter group and the grad table is unambiguous). The QUERY projection
        is shared across branches — all three branches ask the same question, they
        just look at different keys/values.
    """

    def __init__(self, d_model, n_q_heads, n_kv_heads,
                 block_size=16, n_selected_blocks=4, window=32):
        super().__init__()
        assert n_q_heads % n_kv_heads == 0, "GQA needs n_q_heads divisible by n_kv_heads"
        self.d_model = d_model
        self.Hq = n_q_heads
        self.Hkv = n_kv_heads
        self.G = n_q_heads // n_kv_heads          # query heads per kv group
        self.dh = d_model // n_q_heads            # head dim
        self.block_size = block_size
        self.n_sel = n_selected_blocks
        self.window = window
        scale = self.dh ** -0.5
        self.register_buffer("scale", torch.tensor(scale), persistent=False)

        # shared query projection ------------------------------------------------
        self.q_proj = nn.Linear(d_model, self.Hq * self.dh, bias=False)

        # per-branch key/value projections --------------------------------------
        kv_dim = self.Hkv * self.dh
        self.k_cmp = nn.Linear(d_model, kv_dim, bias=False)   # compression
        self.v_cmp = nn.Linear(d_model, kv_dim, bias=False)
        self.k_slc = nn.Linear(d_model, kv_dim, bias=False)   # selection
        self.v_slc = nn.Linear(d_model, kv_dim, bias=False)
        self.k_win = nn.Linear(d_model, kv_dim, bias=False)   # sliding window
        self.v_win = nn.Linear(d_model, kv_dim, bias=False)

        # compression poolers (compression-branch params) ----------------------
        self.comp_k = BlockCompressor(block_size, self.dh)
        self.comp_v = BlockCompressor(block_size, self.dh)

        # per-head, per-branch gate ---------------------------------------------
        # gate(x) -> 3 logits per query head -> sigmoid -> blend weights in [0,1].
        # Bias init 0 => sigmoid 0.5 => ALL three branches active from step 1, so
        # gradient provably flows through every branch on the very first backward.
        self.gate = nn.Linear(d_model, self.Hq * 3, bias=True)

        self.out_proj = nn.Linear(self.Hq * self.dh, d_model, bias=False)

    # -- shape helpers ------------------------------------------------------
    def _split_heads(self, x, n_heads):
        B, T, _ = x.shape
        return x.view(B, T, n_heads, self.dh).transpose(1, 2)   # [B, H, T, dh]

    def forward(self, x):
        B, T, _ = x.shape
        Bsz = self.block_size
        n_blk = T // Bsz                    # number of complete blocks
        Tblk = n_blk * Bsz                  # tokens covered by the block grid
        dev = x.device

        q = self._split_heads(self.q_proj(x), self.Hq)          # [B, Hq, T, dh]

        # =================================================================
        # BRANCH 1 — COMPRESSION
        # =================================================================
        k_c = self._split_heads(self.k_cmp(x), self.Hkv)        # [B, Hkv, T, dh]
        v_c = self._split_heads(self.v_cmp(x), self.Hkv)
        o_cmp = torch.zeros(B, self.Hq, T, self.dh, device=dev, dtype=x.dtype)
        p_cmp = None
        if n_blk > 0:
            # reshape the first Tblk tokens into non-overlapping blocks and pool
            kc_blocks = k_c[:, :, :Tblk].view(B, self.Hkv, n_blk, Bsz, self.dh)
            vc_blocks = v_c[:, :, :Tblk].view(B, self.Hkv, n_blk, Bsz, self.dh)
            k_cmp = self.comp_k(kc_blocks)                      # [B, Hkv, n_blk, dh]
            v_cmp = self.comp_v(vc_blocks)

            k_cmp = expand_kv(k_cmp, self.G)                    # -> [B, Hq, n_blk, dh]
            v_cmp = expand_kv(v_cmp, self.G)

            # scores of each query against each summary token
            s_cmp = torch.matmul(q, k_cmp.transpose(-1, -2)) * self.scale  # [B,Hq,T,n_blk]

            # causal mask on blocks: query t may see summary j only if the whole
            # block precedes-or-includes t, i.e. its LAST token index <= t. This
            # keeps compression strictly causal (no future token leaks via a
            # partially-future block). The block that *contains* t is handled by
            # the sliding-window branch instead.
            t_idx = torch.arange(T, device=dev)[:, None]                  # [T,1]
            blk_end = (torch.arange(n_blk, device=dev) + 1) * Bsz - 1     # [n_blk]
            cmp_keep = (blk_end[None, :] <= t_idx)                        # [T, n_blk]
            cmp_keep = cmp_keep[None, None].expand(B, self.Hq, T, n_blk)

            p_cmp = masked_softmax(s_cmp, cmp_keep)            # [B, Hq, T, n_blk]
            o_cmp = torch.matmul(p_cmp, v_cmp)                 # [B, Hq, T, dh]

        # =================================================================
        # BRANCH 2 — SELECTION  (top-k blocks, scored FROM compression)
        # =================================================================
        o_slc = torch.zeros(B, self.Hq, T, self.dh, device=dev, dtype=x.dtype)
        if n_blk > 0 and p_cmp is not None:
            # ---- block importance, aggregated per GQA group -----------------
            # Because grids are aligned, block j's importance is exactly the
            # compression attention prob on summary token j. NSA shares the block
            # choice across a group, so we SUM the group's query heads.
            imp = p_cmp.view(B, self.Hkv, self.G, T, n_blk).sum(dim=2)    # [B,Hkv,T,n_blk]

            # never pick a non-causal block
            cmp_keep_kv = (blk_end[None, :] <= t_idx)[None, None]          # [1,1,T,n_blk]
            imp = imp.masked_fill(~cmp_keep_kv, torch.finfo(imp.dtype).min)

            # ---- TRAP 1: detach before topk. Indices carry no gradient. -----
            k_top = min(self.n_sel, n_blk)
            sel_idx = torch.topk(imp.detach(), k_top, dim=-1).indices      # [B,Hkv,T,k_top]
            chosen = torch.zeros(B, self.Hkv, T, n_blk, dtype=torch.bool, device=dev)
            chosen.scatter_(-1, sel_idx, True)                            # [B,Hkv,T,n_blk]

            # ---- turn chosen-blocks into a per-token keep mask --------------
            # token i belongs to block i // Bsz (only for i < Tblk).
            blk_of_tok = torch.arange(T, device=dev) // Bsz               # [T]
            blk_of_tok = blk_of_tok.clamp(max=n_blk - 1)
            tok_in_grid = torch.arange(T, device=dev) < Tblk             # [T]
            # gather chosen along the block dim using each token's block id
            sel_keep = chosen[..., blk_of_tok]                           # [B,Hkv,T,T]
            causal = (torch.arange(T, device=dev)[None, :]
                      <= torch.arange(T, device=dev)[:, None])          # [T,T] key<=query
            sel_keep = sel_keep & causal[None, None] & tok_in_grid[None, None, None, :]
            sel_keep = sel_keep.repeat_interleave(self.G, dim=1)         # -> Hq heads

            # ---- attention over the selected fine-grained tokens ------------
            k_s = expand_kv(self._split_heads(self.k_slc(x), self.Hkv), self.G)
            v_s = expand_kv(self._split_heads(self.v_slc(x), self.Hkv), self.G)
            s_slc = torch.matmul(q, k_s.transpose(-1, -2)) * self.scale  # [B,Hq,T,T]
            p_slc = masked_softmax(s_slc, sel_keep)
            o_slc = torch.matmul(p_slc, v_s)                            # gradient -> k_slc/v_slc/q

        # =================================================================
        # BRANCH 3 — SLIDING WINDOW  (recent w tokens)
        # =================================================================
        k_w = expand_kv(self._split_heads(self.k_win(x), self.Hkv), self.G)
        v_w = expand_kv(self._split_heads(self.v_win(x), self.Hkv), self.G)
        s_win = torch.matmul(q, k_w.transpose(-1, -2)) * self.scale     # [B,Hq,T,T]
        qi = torch.arange(T, device=dev)[:, None]
        ki = torch.arange(T, device=dev)[None, :]
        win_keep = (ki <= qi) & (qi - ki < self.window)                # causal band
        win_keep = win_keep[None, None].expand(B, self.Hq, T, T)
        p_win = masked_softmax(s_win, win_keep)
        o_win = torch.matmul(p_win, v_w)                               # [B, Hq, T, dh]

        # =================================================================
        # GATE — per-head sigmoid blend of the three branch outputs
        # =================================================================
        g = self.gate(x).view(B, T, self.Hq, 3)
        g = torch.sigmoid(g).permute(0, 2, 1, 3)                        # [B, Hq, T, 3]
        out = (g[..., 0:1] * o_cmp
               + g[..., 1:2] * o_slc
               + g[..., 2:3] * o_win)                                   # [B, Hq, T, dh]

        out = out.transpose(1, 2).contiguous().view(B, T, self.Hq * self.dh)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# FULL (dense) attention — the matched baseline for rung 2a.
# ---------------------------------------------------------------------------

class FullAttention(nn.Module):
    """Standard causal GQA attention: the control arm of the matched baseline.

    Deliberately the SAME shape as NSA on everything except the attention math:
    identical query heads / kv heads / head dim, identical d_model in and out, a
    shared query projection and single k/v projections, one output projection.
    Kept as full-score + causal mask (same "correct, not fast" style as rung 1) —
    no flash/SDPA fusion, so both arms use the same slow-but-transparent path.

    Only difference vs NSA: NSA has THREE branches (each with its own k/v proj),
    a per-block compressor, and a gate. Those extras are the documented param
    delta between the two configs. Hidden dims are otherwise identical.
    """

    def __init__(self, d_model, n_q_heads, n_kv_heads):
        super().__init__()
        assert n_q_heads % n_kv_heads == 0, "GQA needs n_q_heads divisible by n_kv_heads"
        self.Hq = n_q_heads
        self.Hkv = n_kv_heads
        self.G = n_q_heads // n_kv_heads
        self.dh = d_model // n_q_heads
        self.register_buffer("scale", torch.tensor(self.dh ** -0.5), persistent=False)
        self.q_proj = nn.Linear(d_model, self.Hq * self.dh, bias=False)
        self.k_proj = nn.Linear(d_model, self.Hkv * self.dh, bias=False)
        self.v_proj = nn.Linear(d_model, self.Hkv * self.dh, bias=False)
        self.out_proj = nn.Linear(self.Hq * self.dh, d_model, bias=False)

    def _split_heads(self, x, n_heads):
        B, T, _ = x.shape
        return x.view(B, T, n_heads, self.dh).transpose(1, 2)

    def forward(self, x):
        B, T, _ = x.shape
        q = self._split_heads(self.q_proj(x), self.Hq)                   # [B,Hq,T,dh]
        k = expand_kv(self._split_heads(self.k_proj(x), self.Hkv), self.G)
        v = expand_kv(self._split_heads(self.v_proj(x), self.Hkv), self.G)
        s = torch.matmul(q, k.transpose(-1, -2)) * self.scale           # [B,Hq,T,T]
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        s = s.masked_fill(~causal[None, None], torch.finfo(s.dtype).min)
        p = torch.softmax(s, dim=-1)
        o = torch.matmul(p, v)                                          # [B,Hq,T,dh]
        o = o.transpose(1, 2).contiguous().view(B, T, self.Hq * self.dh)
        return self.out_proj(o)


# ---------------------------------------------------------------------------
# a tiny decoder-only transformer around NSA (or full attention, by flag)
# ---------------------------------------------------------------------------

def make_attention(attn_type, d_model, n_q_heads, n_kv_heads, **nsa_kw):
    """Factory: the ONE place the nsa/full swap happens. Everything else is shared."""
    if attn_type == "nsa":
        return NSAAttention(d_model, n_q_heads, n_kv_heads, **nsa_kw)
    if attn_type == "full":
        return FullAttention(d_model, n_q_heads, n_kv_heads)   # nsa_kw ignored
    raise ValueError(f"unknown attn_type {attn_type!r} (expected 'nsa' or 'full')")


class Block(nn.Module):
    def __init__(self, d_model, n_q_heads, n_kv_heads, ffn_mult=4,
                 attn_type="nsa", **nsa_kw):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = make_attention(attn_type, d_model, n_q_heads, n_kv_heads, **nsa_kw)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = ffn_mult * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class NSATransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=8, n_q_heads=8,
                 n_kv_heads=2, max_seq_len=256, ffn_mult=4,
                 block_size=16, n_selected_blocks=4, window=32,
                 attn_type="nsa"):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.attn_type = attn_type
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_q_heads, n_kv_heads, ffn_mult=ffn_mult,
                  attn_type=attn_type,
                  block_size=block_size, n_selected_blocks=n_selected_blocks,
                  window=window)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight            # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.max_seq_len
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss

    def num_params(self):
        # subtract the tied lm_head (shares tok_emb) to avoid double-counting
        n = sum(p.numel() for p in self.parameters())
        return n - self.lm_head.weight.numel()


# ---------------------------------------------------------------------------
# grad-inspection: map each parameter to a branch so the train loop can print
# a per-branch grad-norm table and assert every branch received gradient.
# ---------------------------------------------------------------------------

# keyword -> branch label. Order matters: first match wins.
_BRANCH_RULES = [
    ("attn.q_proj", "q_proj (shared)"),
    ("attn.k_cmp", "compression"), ("attn.v_cmp", "compression"),
    ("attn.comp_k", "compression"), ("attn.comp_v", "compression"),
    ("attn.k_slc", "selection"), ("attn.v_slc", "selection"),
    ("attn.k_win", "window"), ("attn.v_win", "window"),
    ("attn.gate", "gate"),
    ("attn.out_proj", "out_proj"),
    ("ffn", "ffn"),
    ("tok_emb", "embed"), ("pos_emb", "embed"),
    ("ln", "layernorm"),
]

# the branches we REQUIRE to have live gradient for rung 1 to pass
REQUIRED_BRANCHES = ["q_proj (shared)", "compression", "selection", "window", "gate"]


def branch_of(param_name):
    for key, label in _BRANCH_RULES:
        if key in param_name:
            return label
    return "other"


def param_groups(model):
    """dict: branch label -> list of (name, param)."""
    groups = {}
    for name, p in model.named_parameters():
        groups.setdefault(branch_of(name), []).append((name, p))
    return groups
