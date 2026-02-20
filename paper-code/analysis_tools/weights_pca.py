# =============================
# weights_pca.py — PAPER MODE
# Build PCA/SVD directions on RAW parameter offsets (no whitening), center = best.
# M = [w_e - w*] columns for epochs e < best_epoch
# Top-2 left singular vectors U[:,0:2] are saved as FLAT raw directions.
# Also saves the projected training path XY = U^T @ M (2 x T).
# Defaults: proj_ranges = [-1,1] for both axes (paper).
# =============================
import argparse, os, re, json, importlib, sys, time, shlex, hashlib
from typing import Dict, List, Tuple, Optional
import torch

# repo-root import
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import analysis_tools.compat_case



# --- add this helper near other mapping helpers ---
def _dense_map_with_lora(sd: dict, model_keys: set) -> dict:
    """
    Build a dict of {model_key: tensor} where model_key is in model_keys.
    Fills *.weight/bias from *.base.* and merges LoRA deltas:
        W <- W + (B @ A) * (alpha / r)
    Supports styles: .A/.B, .lora_A/.lora_B, .lora.down/.lora.up
    """
    import torch, re
    mapped = _remap_suffix_keys(sd, model_keys)  # suffix-based first

    # fill from *.base.weight/bias if needed
    for k in list(model_keys):
        if k in mapped: 
            continue
        if k.endswith(".weight"):
            alt = k[:-7] + ".base.weight"
            if alt in sd: mapped[k] = sd[alt]
        elif k.endswith(".bias"):
            alt = k[:-5] + ".base.bias"
            if alt in sd: mapped[k] = sd[alt]

    def _alpha(prefix: str, r: int) -> float:
        for ak in (f"{prefix}.lora_alpha", f"{prefix}.alpha", f"{prefix}.scaling"):
            v = sd.get(ak, None)
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                return float(v.item())
        return float(r)  # safe default

    # patterns: (.lora_A/.lora_B)
    for a_key in [k for k in sd if k.endswith(".lora_A.weight")]:
        prefix = a_key[:-len(".lora_A.weight")]
        b_key  = prefix + ".lora_B.weight"
        base_w = prefix + ".weight"
        if b_key in sd and base_w in mapped:
            A, B = sd[a_key], sd[b_key]
            if getattr(A, "ndim", 0)==2 and getattr(B, "ndim", 0)==2:
                r = int(A.shape[0]); s = _alpha(prefix, r)/max(1,r)
                mapped[base_w] = mapped[base_w] + (B @ A)*s

    # patterns: (.lora.down/.lora.up)
    for d_key in [k for k in sd if k.endswith(".lora.down.weight")]:
        prefix = d_key[:-len(".lora.down.weight")]
        u_key  = prefix + ".lora.up.weight"
        base_w = prefix + ".weight"
        if u_key in sd and base_w in mapped:
            A, B = sd[d_key], sd[u_key]
            if getattr(A, "ndim", 0)==2 and getattr(B, "ndim", 0)==2:
                r = int(A.shape[0]); s = _alpha(prefix, r)/max(1,r)
                mapped[base_w] = mapped[base_w] + (B @ A)*s

    # patterns: (.A/.B)
    for a_key in [k for k in sd if k.endswith(".A")]:
        prefix = a_key[:-len(".A")]
        b_key  = prefix + ".B"
        base_w = prefix + ".weight"
        if b_key in sd and base_w in mapped:
            A, B = sd[a_key], sd[b_key]
            if getattr(A, "ndim", 0)==2 and getattr(B, "ndim", 0)==2:
                r = int(A.shape[0]); s = _alpha(prefix, r)/max(1,r)
                mapped[base_w] = mapped[base_w] + (B @ A)*s

    return mapped


def iter_param_groups(model, include_frozen: bool = False):
    """
    Yield (name, param) over model parameters.
    Default: trainable only (include_frozen=False) to match 'save everything' setups.
    """
    for name, p in model.named_parameters():
        if not include_frozen and (not p.requires_grad):
            continue
        yield name, p

# ----- args-file parser -----
class CommentArgParser(argparse.ArgumentParser):
    def __init__(self, *a, **kw):
        kw.setdefault("fromfile_prefix_chars", "@")
        super().__init__(*a, **kw)
    def convert_arg_line_to_args(self, line: str):
        line = line.strip()
        if not line or line.startswith("#"): return []
        return [tok for tok in shlex.split(line, comments=True) if tok]

def _xp(p): return os.path.expanduser(os.path.expandvars(p)) if p else p

def paper_normalize(vec_flat, slices, model):
    v = vec_flat.clone(); pmap = dict(model.named_parameters()); eps = 1e-10
    with torch.no_grad():
        for name, a, b, shape in slices:
            if len(shape) <= 1: v[a:b].zero_(); continue
            seg = v[a:b]; dn = float(seg.norm().clamp_min(eps))
            wn = float(pmap[name].data.norm().clamp_min(eps))
            v[a:b] = seg * (wn / dn)
    return v


def load_factory(path: str):
    m,f = path.split(':')
    return getattr(importlib.import_module(m), f)

# ----- run-dir helpers -----
_NUM_RE = re.compile(r'(\d+)')

def _find_metrics_path(run_dir: str) -> Optional[str]:
    m1 = os.path.join(run_dir, 'metrics.json')
    if os.path.isfile(m1): return m1
    for f in os.listdir(run_dir):
        if f.endswith('-metrics.json'): return os.path.join(run_dir, f)
    return None

def _maybe_best_from_metrics(metrics_path: str) -> Optional[int]:
    try:
        with open(metrics_path, 'r') as f:
            m = json.load(f)
        if isinstance(m, dict):
            if 'best_epoch' in m: return int(m['best_epoch'])
            if isinstance(m.get('best'), dict) and 'epoch' in m['best']: return int(m['best']['epoch'])
            if 'best_ckpt' in m:
                nums = _NUM_RE.findall(str(m['best_ckpt']))
                if nums: return int(nums[-1])
    except Exception:
        pass
    return None

def _find_ckpts(run_dir: str) -> List[str]:
    cand = []
    for root, _, files in os.walk(run_dir):
        for f in files:
            fn = f.lower()
            if fn.endswith(('.pt','.pth','.bin')) and 'optim' not in fn and 'optimizer' not in fn:
                cand.append(os.path.join(root,f))
    def _key(p):
        parts = re.split(r'(\d+)', os.path.basename(p))
        return [int(t) if t.isdigit() else t for t in parts]
    return sorted(set(cand), key=_key)

def _parse_epoch_from_name(path: str) -> Optional[int]:
    name = os.path.basename(path).lower()
    if "best" in name: return None
    pats = [
        r'(?:^|[_\-])(?:epoch|ep|e)[=_\-]?(\d+)(?:[^0-9]|$)',
        r'(?:^|[_\-])(?:checkpoint|ckpt)[=_\-]?(\d+)(?:[^0-9]|$)',
        r'(\d+)(?:\.pt|\.pth|\.bin)$',
    ]
    for pat in pats:
        m = re.search(pat, name)
        if m:
            try: return int(m.group(1))
            except Exception: return None
    return None

def _select_ckpts_to_best(paths: List[str], best_epoch: Optional[int]):
    # filter out explicit "best" files
    paths = [p for p in paths if "best" not in os.path.basename(p).lower()]
    items = []
    for p in paths:
        e = _parse_epoch_from_name(p)
        try: mt = os.path.getmtime(p)
        except Exception: mt = 0.0
        items.append((e,p,mt))
    if all(e is None for e,_,_ in items):
        items.sort(key=lambda t: t[2])
        return [t[1] for t in items[:-1]], items[-1][1], None  # newest as "best"
    usable = [t for t in items if t[0] is not None]
    if best_epoch is None:
        best_epoch = max(t[0] for t in usable)
    best_cands = [t for t in usable if t[0] == best_epoch]
    best_path = max(best_cands, key=lambda t: t[2])[1] if best_cands else None
    cols = [t for t in usable if t[0] < best_epoch]
    newest_by_epoch = {}
    for e, p, mt in cols:
        cur = newest_by_epoch.get(e)
        if (cur is None) or (mt > cur[1]):
            newest_by_epoch[e] = (p, mt)
    chosen_paths = [newest_by_epoch[e][0] for e in sorted(newest_by_epoch)]
    return chosen_paths, best_path, best_epoch

# ----- state dict alignment -----
def _safe_torch_load(path, map_location='cpu'):
    try:  return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError: return torch.load(path, map_location=map_location)

def _load_state_dict_candidates(obj) -> Dict[str, torch.Tensor]:
    if isinstance(obj, str):
        sd = _safe_torch_load(obj, map_location='cpu')
    else:
        sd = obj
    if isinstance(sd, dict):
        for k in ['state_dict','model_state_dict','module','model','weights']:
            if k in sd and isinstance(sd[k], dict):
                return sd[k]
        if all(isinstance(v, torch.Tensor) for v in sd.values()):
            return sd
    raise RuntimeError(f"Could not find a state_dict mapping in {obj}")

def _remap_suffix_keys(sd: dict, model_keys: set):
    mapped = {}
    for k, v in sd.items():
        parts = k.split(".")
        for i in range(len(parts)):
            suf = ".".join(parts[i:])
            if suf in model_keys:
                mapped[suf] = v
                break
    return mapped

def flatten_with_map(named_params):
    flat, slices, off = [], [], 0
    for name, t in named_params:
        n = t.numel()
        flat.append(t.detach().reshape(-1))
        slices.append((name, off, off+n, tuple(t.shape)))
        off += n
    return torch.cat(flat), slices

def _align_flat_from_sd(sd: Dict[str, torch.Tensor], named) -> torch.Tensor:
    targets = {n for n,_ in named}
    sd_map = _dense_map_with_lora(sd, targets)  # <-- use the new helper

    parts, missing = [], 0
    for name, p in named:
        t = sd_map.get(name)
        if t is None:
            parts.append(p.detach().reshape(-1).cpu())  # fallback keeps shape
            missing += 1
        else:
            parts.append(t.detach().reshape(-1).cpu())
    if missing:
        print(f"[PCA][INFO] Filled {missing} missing slices from current model state.")
    return torch.cat(parts) if parts else torch.empty(0)

# ----- main -----
def main():
    ap = CommentArgParser(description='RAW PCA/SVD directions (paper mode) + projected path.', fromfile_prefix_chars='@')
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--factory', required=True)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--scope', default='trainable', help='(ignored) kept for backward-compat')
    ap.add_argument('--include-frozen', type=lambda s: s.lower()!='false', default=False)
    ap.add_argument('--limit', type=int, default=None, help='Max # checkpoints (latest first, still <= best).')
    ap.add_argument('--print-every', type=int, default=5)
    ap.add_argument('--center-epoch', type=int, default=None,
                    help='Override: use this epoch as the center (w*), only use epochs < center.')
    ap.add_argument('--center-by', choices=['auto','loss'], default='auto',
                    help='How to choose center if --center-epoch is not set. '
                         '"auto" = metrics.json best; "loss" = min val_loss from epochs/*.json.')
    args = ap.parse_args()

    args.run_dir = _xp(args.run_dir); args.out_dir = _xp(args.out_dir)
    t0 = time.time()
    default_root = "landscape_pca"  # paper-mode default directory
    out_dir = args.out_dir or os.path.join(args.run_dir, default_root)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[PCA] Run dir: {args.run_dir}")
    print(f"[PCA] Output dir: {out_dir}")
    print(f"[PCA] Mode: PAPER (raw offsets SVD), scope={args.scope}, include_frozen={args.include_frozen}")

    build = load_factory(args.factory)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[PCA] Building model on device={device} (for param layout only) ...")
    model, _ = build(run_dir=args.run_dir, device=device)
    named = list(iter_param_groups(model, include_frozen=args.include_frozen))
    base_flat, slices = flatten_with_map(named)
    dims = base_flat.numel()
    print(f"[PCA] Parameters discovered: tensors={len(slices)} dims={dims:,}")

    # choose checkpoints (to best)
    def _best_epoch_by_loss(run_dir: str):
        """Scan epochs/*.json and return (epoch, val_loss, checkpoint_path) with min loss."""
        epochs_dir = os.path.join(run_dir, 'epochs')
        best = None
        try:
            for f in os.listdir(epochs_dir):
                if not f.lower().endswith('.json'):
                    continue
                try:
                    with open(os.path.join(epochs_dir, f), 'r') as jf:
                        j = json.load(jf)
                    e = int(j.get('epoch'))
                    loss = j.get('val_loss', None)
                    ckpt = j.get('checkpoint_path', None)
                    if loss is None:
                        continue
                    if (best is None) or (loss < best[1]):
                        best = (e, float(loss), ckpt)
                except Exception:
                    continue
        except Exception:
            pass
        return best  # or None

    metrics_path = _find_metrics_path(args.run_dir)
    best_epoch_hint = _maybe_best_from_metrics(metrics_path) if metrics_path else None
    center_epoch = args.center_epoch
    center_path_override = None
    if center_epoch is None and args.center_by == 'loss':
        los = _best_epoch_by_loss(args.run_dir)
        if los is not None:
            center_epoch, _, center_path_override = los
    if center_epoch is None:
        center_epoch = best_epoch_hint

    ckpts_all = _find_ckpts(args.run_dir)
    chosen, best_path, best_epoch = _select_ckpts_to_best(ckpts_all, center_epoch)
    if center_path_override:
        best_path = center_path_override
        best_epoch = center_epoch
    if args.limit is not None and args.limit > 0:
        chosen = chosen[-args.limit:]
        print(f"[PCA] Limiting to last {len(chosen)} checkpoints before best.")
    if not best_path:
        print("[PCA][WARN] Could not identify a concrete 'best' checkpoint; using newest as center if needed.")

    print(f"[PCA] best_epoch={best_epoch} best_path={os.path.basename(best_path) if best_path else '<?>'}")
    print(f"[PCA] Matrix columns (epochs < best): {len(chosen)}")

    # align flats
    target_names = [n for n,_ in named]
    def align(path):
        sd = _load_state_dict_candidates(path)
        vec = _align_flat_from_sd(sd, named)
        if vec is None:
            raise RuntimeError(f"Failed to align {os.path.basename(path)} to current parameter layout.")
        return vec

    W_best = align(best_path) if best_path else align(ckpts_all[-1])
    cols = []
    for i, p in enumerate(chosen):
        if (i % max(1, args.print_every)) == 0:
            print(f"[PCA] loading/aligining [{i+1}/{len(chosen)}]: {os.path.basename(p)}")
        Wi = align(p)
        cols.append((Wi - W_best).to(torch.float32))  # RAW offsets (no whitening)
    if not cols:
        raise RuntimeError("No checkpoints < best to build PCA matrix.")
    # Build D x K matrix with columns offsets
    M = torch.stack(cols, dim=1)  # [D, K]
    D, K = M.shape
    print(f"[PCA] Offset matrix M shape: {D:,} x {K}")

    # economical SVD via KxK Gram (since K << D)
    # G = M^T M  => eig(G) gives singular values^2 and right vecs; left vecs U = M @ V / s
    print("[PCA] Computing top-2 left singular vectors via Gram eigen-decomposition ...")
    G = M.T @ M  # [K,K]
    evals, V = torch.linalg.eigh(G)  # ascending
    idx = torch.argsort(evals, descending=True)
    evals = evals[idx]; V = V[:, idx]  # [K,K]
    s = evals.clamp(min=0).sqrt()      # singular values

    if s[0] <= 0:
        raise RuntimeError("Top singular value is non-positive; degenerate matrix?")
    k_use = min(2, V.shape[1])
    V_k = V[:, :k_use]                 # [K,2]
    s_k = s[:k_use]                    # [2]
    U_k = (M @ V_k) / s_k.unsqueeze(0) # [D,2]

    v1 = U_k[:,0].contiguous()
    v2 = U_k[:,1] if k_use > 1 else torch.randn(D)

    # meta/fingerprint
    names_shapes = [(n, tuple(int(x) for x in shape)) for n,_,_,shape in slices]
    fp = hashlib.sha1(repr(names_shapes).encode()).hexdigest()[:16]
    meta = {
        'scope': args.scope,
        'basis': 'paper_raw',
        'proj_ranges': {'x':(-1.0,1.0), 'y':(-1.0,1.0)},
        'fingerprint': fp,
        'best_ckpt': os.path.basename(best_path) if best_path else None,
        'best_epoch': int(best_epoch) if best_epoch is not None else None,
    }
    
    # paper-normalize the axes using the SAME model we just built
    v1p = paper_normalize(v1, slices, model)
    v2p = paper_normalize(v2, slices, model)
    Vp  = torch.stack([v1p, v2p], dim=1)         # [D,2]
    G2  = Vp.T @ Vp                              # 2x2 Gram
    B   = torch.linalg.pinv(G2)                   # (G2)^{-1}

    # project each offset x onto the (non-orthonormal) paper-normalized plane:
    # c = (G2^{-1}) @ (Vp^T x)
    W_all = torch.stack([align(p) for p in chosen], dim=1)   # [D, K]
    M_all = W_all - W_best.unsqueeze(1)                      # [D, K]
    XY_paper = (B @ (Vp.T @ M_all)).cpu()                    # [2, K]

    torch.save({'XY': XY_paper, 'coords':'paper'}, os.path.join(out_dir, 'path_xy.pt'))


    torch.save({**meta, 'v': v1.cpu()}, os.path.join(out_dir, 'dirX.pth'))
    torch.save({**meta, 'v': v2.cpu()}, os.path.join(out_dir, 'dirY.pth'))
    print(f"[PCA] Wrote directions: {out_dir}/dirX.pth & dirY.pth (RAW, paper mode).")

    # Project full trajectory (XY = U^T @ (W - W_best))
    #W_all = torch.stack([align(p) for p in chosen], dim=1)  # [D, K]
    #M_all = W_all - W_best.unsqueeze(1)
    #U2 = torch.stack([v1, v2], dim=1)  # [D,2]
    #XY = U2.T @ M_all                  # [2, K]
    #torch.save({'XY': XY.cpu()}, os.path.join(out_dir, 'path_xy.pt'))
    print(f"[PCA] Saved projected path (2 x {XY_paper.shape[1]}) to {out_dir}/path_xy.pt")

    print(f"[PCA] Done in {time.time()-t0:.1f}s.")

if __name__ == '__main__':
    main()
