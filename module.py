import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    """AdaLN-zero modulation."""
    return x * (1 + scale) + shift


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer."""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """Compute SIGReg over projections shaped ``(T, B, D)``."""
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.cond_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.output_proj = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()
        for _ in range(depth):
            self.layers.append(block_class(hidden_dim, heads, dim_head, mlp_dim, dropout))

    def forward(self, x, c=None):
        x = self.input_proj(x)
        c = self.cond_proj(c) if c is not None else c
        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)
        return self.output_proj(x)


class Embedder(nn.Module):
    def __init__(self, input_dim=10, smoothed_dim=10, emb_dim=10, mlp_scale=4):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        return self.embed(x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, norm_fn=nn.LayerNorm, act_fn=nn.GELU):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        return self.net(x)


class GaussianMLP(nn.Module):
    """Predictor head that outputs diagonal-Gaussian parameters (mu, log_var)."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
        min_log_var: float = -6.0,
        max_log_var: float = 6.0,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.min_log_var = float(min_log_var)
        self.max_log_var = float(max_log_var)
        self.backbone = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=2 * output_dim,
            norm_fn=norm_fn,
            act_fn=act_fn,
        )

    def forward(self, x):
        params = self.backbone(x)
        mu, log_var = params.chunk(2, dim=-1)
        log_var = log_var.clamp(self.min_log_var, self.max_log_var)
        return mu, log_var


class QuasimetricHead(nn.Module):
    """Asymmetric latent distance d(z_s, z_g) calibrated to temporal reachability.

    Metric-Residual-Network style: a shared projection splits into a symmetric
    Euclidean component and an asymmetric max-ReLU component. The result is
    non-negative and asymmetric in the (state, goal) order, so it can capture
    directed steps-to-goal (control reachability is not symmetric). Trained to
    regress the temporal gap between a state and a future state.
    """

    def __init__(self, input_dim, hidden_dim=512, sym_dim=128, asym_dim=128):
        super().__init__()
        self.sym_dim = sym_dim
        self.asym_dim = asym_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, sym_dim + asym_dim),
        )

    def project(self, z):
        feats = self.backbone(z)
        return feats[..., : self.sym_dim], feats[..., self.sym_dim :]

    def forward_parts(self, z_s, z_g):
        """Return symmetric and asymmetric distance components."""
        sym_s, asym_s = self.project(z_s)
        sym_g, asym_g = self.project(z_g)
        sym = (sym_s - sym_g).pow(2).sum(dim=-1).clamp_min(1e-12).sqrt()
        asym = torch.relu(asym_s - asym_g).amax(dim=-1)
        return sym, asym

    def forward(self, z_s, z_g):
        """Return directed distance d(z_s -> z_g) with shape ``z_s.shape[:-1]``."""
        sym, asym = self.forward_parts(z_s, z_g)
        return sym + asym


class EuclideanDistanceHead(nn.Module):
    """Symmetric temporal-distance head for ablation against directed costs.

    The head maps latents into a learned metric space and returns Euclidean
    distance. It uses the same ``distance(z_s, z_g)`` interface as
    ``QuasimetricHead`` but removes the asymmetric residual, isolating whether
    directionality is useful beyond temporal-distance supervision itself.
    """

    def __init__(self, input_dim, hidden_dim=512, metric_dim=128):
        super().__init__()
        self.metric_dim = metric_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, metric_dim),
        )

    def project(self, z):
        return self.backbone(z)

    def forward_parts(self, z_s, z_g):
        """Return symmetric distance and a zero asymmetric component."""
        feat_s = self.project(z_s)
        feat_g = self.project(z_g)
        sym = (feat_s - feat_g).pow(2).sum(dim=-1).clamp_min(1e-12).sqrt()
        return sym, sym.new_zeros(sym.shape)

    def forward(self, z_s, z_g):
        """Return symmetric Euclidean distance with shape ``z_s.shape[:-1]``."""
        sym, _ = self.forward_parts(z_s, z_g)
        return sym


class ReachabilityHead(nn.Module):
    """Budget-conditioned reachability classifier for RC-aux.

    Scores whether goal latent ``z_g`` is reachable from ``z_s`` within the
    planner horizon. Non-directed: inputs are concatenated and mapped to a
    single logit (no quasimetric residual).
    """

    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z_s, z_g):
        """Return reachability logits with shape ``z_s.shape[:-1]``."""
        logits = self.backbone(torch.cat([z_s, z_g], dim=-1))
        return logits.squeeze(-1)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        return self.transformer(x, c)
