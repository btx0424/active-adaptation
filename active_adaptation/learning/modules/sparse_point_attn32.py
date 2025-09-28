from typing import List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------- helpers ---------------------
def _fold_to_batch(x: torch.Tensor, data_dims: int) -> Tuple[torch.Tensor, int]:
    """Fold all leading dims except the last `data_dims` into batch.
    Returns (x_flat, N_leading) where N_leading = product of leading dims.
    """
    assert x.ndim >= data_dims, f"tensor has {x.ndim} dims, needs >= {data_dims}"
    if x.ndim == data_dims:
        return x, 1
    N = 1
    for s in x.shape[:-data_dims]:
        N *= int(s)
    x = x.reshape(N, *x.shape[-data_dims:])
    return x, N

def _dense32_to_points(vox_b1d32: torch.Tensor, thresh: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """vox_b1d32: [N,1,32,32,32] float
    return pts_bxyz: [M,4] long (b,z,y,x), feats: [M,1] float
    """
    assert vox_b1d32.ndim == 5 and vox_b1d32.shape[1:] == (1,32,32,32), f"got {vox_b1d32.shape}"
    N = vox_b1d32.size(0)
    v = vox_b1d32.squeeze(1)  # [N,32,32,32]

    b_cols, z_cols, y_cols, x_cols = [], [], [], []
    for b in range(N):
        idx = (v[b] > thresh).nonzero(as_tuple=False)  # [n,3]
        if idx.numel() == 0:
            idx = torch.tensor([[0,0,0]], device=v.device, dtype=torch.long)
        n = idx.size(0)
        b_cols.append(torch.full((n,1), b, device=v.device, dtype=torch.long))
        z_cols.append(idx[:,0:1]); y_cols.append(idx[:,1:2]); x_cols.append(idx[:,2:3])

    b_all = torch.cat(b_cols, 0); z_all = torch.cat(z_cols, 0)
    y_all = torch.cat(y_cols, 0); x_all = torch.cat(x_cols, 0)
    pts_bxyz = torch.cat([b_all, z_all, y_all, x_all], dim=1)  # [M,4]
    feats = torch.ones((pts_bxyz.size(0), 1), device=v.device, dtype=torch.float32)
    return pts_bxyz, feats

def _normalize_xyz32(pts_bxyz: torch.Tensor) -> torch.Tensor:
    z = pts_bxyz[:,1].float() / 31.0
    y = pts_bxyz[:,2].float() / 31.0
    x = pts_bxyz[:,3].float() / 31.0
    return torch.stack([x,y,z], dim=1)  # [M,3]

# ---------------- neighborhood attention (windowed) ----------------
class NeighborhoodAttention(nn.Module):
    """Per-window KNN multi-head attention.
    Uses q/kv/proj Linear (bias=False for q/kv), LayerNorm handled outside.
    """
    def __init__(self, dim: int, heads: int = 4, k: int = 16, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} must be divisible by heads {heads}"
        self.dim, self.h, self.k = dim, heads, k
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim*2, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def _attend_group(self, f: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Attention inside one (env, window) group.
        f: [n,C], p: [n,3] in [0,1]
        """
        n, C = f.shape
        if n == 0:
            return f
        H = self.h; Dh = C // H

        # KNN（fp32 距离以稳）
        with torch.no_grad():
            d = torch.cdist(p.float(), p.float())
            k_used = int(min(self.k, n))
            knn = d.topk(k=k_used, largest=False).indices  # [n,k_used]

        q = self.q(f); kv = self.kv(f); k, v = kv.chunk(2, dim=1)
        qh = q.view(n, H, Dh); kh = k.view(n, H, Dh); vh = v.view(n, H, Dh)

        gather_idx = knn.reshape(-1)
        kh_n = kh.index_select(0, gather_idx).view(n, k_used, H, Dh)
        vh_n = vh.index_select(0, gather_idx).view(n, k_used, H, Dh)

        att = (qh.unsqueeze(1) * kh_n).sum(-1) / math.sqrt(Dh)  # [n,k_used,H]
        att = F.softmax(att, dim=1)
        att = self.drop(att)

        o = (att.unsqueeze(-1) * vh_n).sum(1).reshape(n, C)     # [n,C]
        return self.proj(o)

    def forward(self, feats: torch.Tensor, xyz: torch.Tensor,
                bids: torch.Tensor, win_id: torch.Tensor, G3: int) -> torch.Tensor:
        """Windowed attention across groups defined by (bids, win_id).
        feats: [M,C] (dtype could be fp16 in autocast)
        xyz:   [M,3]
        bids:  [M] env id in [0..N-1]
        win_id:[M] window id in [0..G3-1], where G3 = (32//w)^3
        """
        out = torch.empty_like(feats)
        key = bids.to(torch.int64) * G3 + win_id.to(torch.int64)  # [M]

        order = torch.argsort(key)  # group by key
        feats = feats[order]; xyz = xyz[order]; key = key[order]

        # find group boundaries
        if key.numel() == 0:
            return feats  # nothing to do
        diff = torch.ones_like(key)
        diff[1:] = (key[1:] != key[:-1]).to(diff.dtype)
        starts = (diff.nonzero(as_tuple=False).squeeze(1)).tolist()
        starts.append(key.numel())

        for si in range(len(starts)-1):
            a = starts[si]; b = starts[si+1]
            f = feats[a:b]; p = xyz[a:b]
            src = self._attend_group(f, p)
            # AMP-safe: align dtype to out
            if src.dtype != out.dtype:
                src = src.to(out.dtype)
            out[order[a:b]] = src
        return out

# ---------------------- encoder ----------------------
class MixedEncoder(nn.Module):
    def __init__(self, mlp_out=256, cnn_out=32, conv3d: bool=False,
                 width: int = 64, heads: int = 4, k: int = 12, layers: int = 1,
                 thresh: float = 0.6, dropout: float = 0.0,
                 window_size: int = 4, max_points_per_env: int = 3000):
        """
        conv3d=True : fast sparse point+windowed-attn encoder
        conv3d=False: original-style light 2D CNN path
        """
        super().__init__()
        self.mlp_out = mlp_out
        self.cnn_out = cnn_out
        self.conv3d = conv3d
        self.thresh = thresh
        self.width = width
        self.window_size = int(window_size)
        assert 32 % self.window_size == 0, "window_size must divide 32"
        self.G = 32 // self.window_size
        self.G3 = self.G * self.G * self.G
        self.max_points_per_env = max_points_per_env

        # MLP branch
        self.mlp_encoder = nn.Sequential(
            nn.LazyLinear(256), nn.SiLU(), nn.LayerNorm(256),
            nn.LazyLinear(256)
        )

        if not conv3d:
            # 2D fallback
            self.cnn_encoder_2d = nn.Sequential(
                nn.LazyConv2d(8, kernel_size=3, stride=2, padding=1), nn.SiLU(),
                nn.LazyConv2d(8, kernel_size=3, stride=2, padding=1), nn.SiLU(),
                nn.LazyConv2d(8, kernel_size=3, stride=2, padding=1), nn.SiLU(),
                nn.Flatten(),
                nn.LazyLinear(32), nn.SiLU(), nn.LayerNorm(32),
                nn.LazyLinear(256),
            )
        else:
            # sparse windowed attention
            self.inp = nn.Linear(1, width)
            blocks = []
            for _ in range(layers):
                blocks += [
                    NeighborhoodAttention(width, heads=heads, k=k, dropout=dropout),
                    nn.LayerNorm(width), nn.SiLU(),
                    nn.Linear(width, width), nn.LayerNorm(width), nn.SiLU(),
                ]
            self.blocks = nn.ModuleList(blocks)
            self.readout = nn.Sequential(
                nn.Linear(width, 128), nn.SiLU(), nn.LayerNorm(128),
                nn.Linear(128, 256)
            )

        self.out = nn.Sequential(nn.SiLU(), nn.LazyLinear(256), nn.SiLU())

    @staticmethod
    def _canon_last4(x_last4: torch.Tensor) -> torch.Tensor:
        """x_last4: either [1,32,32,32] or [32,1,32,32] -> [1,32,32,32]."""
        if x_last4.shape == (1,32,32,32):
            return x_last4
        if x_last4.shape == (32,1,32,32):
            return x_last4.permute(1,0,2,3).contiguous()
        raise ValueError(f"unsupported last-4 shape {x_last4.shape}")

    def forward(self, mlp_inp: torch.Tensor, cnn_inp: torch.Tensor, mask_cnn: torch.Tensor=None):
        # 1) MLP first to get leading shape
        mlp_feature = self.mlp_encoder(mlp_inp)  # [..., 256]
        target_leading = mlp_feature.shape[:-1]
        N_target = 1
        for s in target_leading: N_target *= int(s)

        if not self.conv3d:
            # --- 2D path ---
            x2d, N_lead = _fold_to_batch(cnn_inp.float(), data_dims=3)   # [N, C, H, W]
            cnn_feature = self.cnn_encoder_2d(x2d)                        # [N, 256]
            if cnn_feature.size(0) != N_target:
                raise RuntimeError(f"cnn_feature N={cnn_feature.size(0)} != prod(mlp leading)={N_target}")
            cnn_feature = cnn_feature.view(*target_leading, 256)
        else:
            # --- 3D sparse+windowed attention path ---
            x3d, N_lead = _fold_to_batch(cnn_inp, data_dims=4)            # [N, C_orZ, D_or1, H, W]
            last4 = x3d.shape[-4:]
            if last4 == (1,32,32,32):
                vox = x3d
            elif last4 == (32,1,32,32):
                vox = x3d.permute(0,2,1,3,4).contiguous()
            else:
                raise ValueError(f"Unexpected voxel tail shape {last4}, expected (1,32,32,32) or (32,1,32,32)")
            vox = vox.float()  # [N,1,32,32,32]

            # points
            pts_bxyz, feats = _dense32_to_points(vox, self.thresh)  # [M,4], [M,1]
            bids = pts_bxyz[:,0].long()
            xyz  = _normalize_xyz32(pts_bxyz)                       # [M,3]

            # (a) per-env cap
            if self.max_points_per_env is not None:
                keep = torch.zeros(bids.numel(), dtype=torch.bool, device=bids.device)
                N_fold = vox.size(0)
                for b in range(N_fold):
                    idx = (bids == b).nonzero(as_tuple=False).squeeze(1)
                    nb = idx.numel()
                    if nb == 0: continue
                    if nb > self.max_points_per_env:
                        perm = torch.randperm(nb, device=bids.device)[:self.max_points_per_env]
                        idx = idx[perm]
                    keep[idx] = True
                if not keep.all():
                    pts_bxyz = pts_bxyz[keep]; feats = feats[keep]
                    bids = bids[keep]; xyz = xyz[keep]

            # (b) window id
            w = self.window_size; G = self.G; G3 = self.G3
            q = (xyz * 31.0).round().clamp_(0,31).to(torch.int32)  # [M,3]
            win = q // w                                           # [M,3] in [0..G-1]
            win_id = (win[:,0] + win[:,1]*G + win[:,2]*G*G).to(torch.int64)  # [M]

            # features
            x = self.inp(feats)  # [M, width]
            for i in range(0, len(self.blocks), 6):
                attn, ln1, act1, ffn, ln2, act2 = self.blocks[i:i+6]
                res = x
                x = attn(x, xyz, bids, win_id, G3) + res
                x = act1(ln1(x))
                x = x + act2(ln2(ffn(x)))

            # pooled per folded env -> [N, width]
            N_fold = vox.size(0)
            pooled = torch.zeros((N_fold, x.size(1)), device=x.device, dtype=x.dtype)
            if bids.numel() > 0:
                pooled.index_add_(0, bids, x)
                cnt = torch.bincount(bids, minlength=N_fold).clamp_min(1).unsqueeze(1).to(x.dtype)
                pooled = pooled / cnt
            cnn_feature = self.readout(pooled)  # [N,256]

            if cnn_feature.size(0) != N_target:
                raise RuntimeError(f"cnn_feature N={cnn_feature.size(0)} != prod(mlp leading)={N_target}")
            cnn_feature = cnn_feature.view(*target_leading, 256)

        if mask_cnn is not None:
            cnn_feature = cnn_feature * mask_cnn

        feature = mlp_feature + cnn_feature
        return self.out(feature)

# --------------------- safe init ---------------------
def safe_init_(m: nn.Module):
    """Initialize while skipping Lazy / None biases safely."""
    from torch.nn.parameter import UninitializedParameter
    if isinstance(m, nn.Linear):
        if isinstance(m.weight, UninitializedParameter):
            return
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        b = getattr(m, "bias", None)
        if b is not None and not isinstance(b, UninitializedParameter):
            nn.init.zeros_(b)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        if isinstance(m.weight, UninitializedParameter):
            return
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        b = getattr(m, "bias", None)
        if b is not None and not isinstance(b, UninitializedParameter):
            nn.init.zeros_(b)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        w = getattr(m, "weight", None)
        if w is not None and not isinstance(w, UninitializedParameter):
            nn.init.ones_(w)
        b = getattr(m, "bias", None)
        if b is not None and not isinstance(b, UninitializedParameter):
            nn.init.zeros_(b)
    else:
        return

# --------------------- self-test ---------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Fake inputs with leading dims [E=4, T=48]
    E, T = 4, 48
    mlp_inp = torch.randn(E, T, 128, device=device)
    vox = (torch.rand(E, T, 32, 1, 32, 32, device=device) < 0.02).float()  # [E,T,32,1,32,32]

    enc = MixedEncoder(conv3d=True, width=32, heads=4, k=12, layers=1,
                       thresh=0.6, window_size=4, max_points_per_env=3000).to(device)

    # materialize lazy layers then init
    out = enc.mlp_encoder(mlp_inp)
    enc.apply(safe_init_)

    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16):
        y = enc(mlp_inp, vox, None)
    print("Output:", tuple(y.shape))  # expect [E,T,256]