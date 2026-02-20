# =============================
# plot_2d_with_path.py — PAPER MODE
# 2D contour + projected path using paper-style normalization & [-1,1] ranges.
# - Directions are expected as raw PCA/SVD vectors (flat) saved by weights_pca.py
# - We paper-normalize them against the CURRENT model before evaluating loss:
#     * scale each per-parameter slice so ||d_k|| = ||W_k||
#     * zero any 1D tensors (bias, LN scales, etc.)
# - Loss is computed on the VALIDATION loader from your factory.
# - Path overlay reads <dir>/path_xy.pt (2 x T) if present.
# =============================
from collections.abc import Mapping
import argparse, os, importlib, sys, time, shlex, hashlib
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import analysis_tools.compat_case  # enables case-insensitive @file includes

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import json, re

_NUM_RE = re.compile(r'(\d+)')

def _find_metrics_path(run_dir: str):
    m1 = os.path.join(run_dir, 'metrics.json')
    if os.path.isfile(m1): return m1
    for f in os.listdir(run_dir):
        if f.endswith('-metrics.json'):
            return os.path.join(run_dir, f)
    return None

def _maybe_best_from_metrics(metrics_path: str):
    try:
        with open(metrics_path, 'r') as f:
            m = json.load(f)
        if isinstance(m, dict):
            if 'best_epoch' in m: return int(m['best_epoch'])
            if isinstance(m.get('best'), dict) and 'epoch' in m['best']:
                return int(m['best']['epoch'])
            if 'best_ckpt' in m:
                nums = _NUM_RE.findall(str(m['best_ckpt']))
                if nums: return int(nums[-1])
    except Exception:
        pass
    return None

def _best_epoch_by_loss(run_dir: str):
    """Return (epoch, val_loss, checkpoint_path_or_None) for min val_loss.
       Prefers epochs/*.json; falls back to run_dir/metrics_by_epoch.csv."""
    best = None
    epochs_dir = os.path.join(run_dir, 'epochs')
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
    if best is None:
        csv_path = os.path.join(run_dir, 'metrics_by_epoch.csv')
        if os.path.isfile(csv_path):
            import csv
            with open(csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        e = int(row.get('epoch'))
                        loss = float(row.get('val_loss'))
                    except Exception:
                        continue
                    if (best is None) or (loss < best[1]):
                        best = (e, loss, None)
    return best  # or None

def _find_ckpts(run_dir: str):
    cand = []
    for root, _, files in os.walk(run_dir):
        for f in files:
            fn = f.lower()
            if fn.endswith(('.pt','.pth','.bin')) and 'optim' not in fn and 'optimizer' not in fn:
                cand.append(os.path.join(root, f))
    def _key(p):
        parts = re.split(r'(\d+)', os.path.basename(p))
        return [int(t) if t.isdigit() else t for t in parts]
    return sorted(set(cand), key=_key)

def _parse_epoch_from_name(path: str):
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

def _choose_ckpt_for_epoch(run_dir: str, epoch: int):
    """Pick the newest checkpoint file whose epoch == given epoch."""
    paths = _find_ckpts(run_dir)
    usable = [(p, _parse_epoch_from_name(p)) for p in paths]
    cands = [p for p,e in usable if e == epoch]
    if not cands:
        return None
    # newest mtime among the matching epoch
    cands = sorted(cands, key=lambda p: os.path.getmtime(p))
    return cands[-1]

def _load_state_dict_candidates(obj):
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

def _dense_map_with_lora(sd: dict, model_keys: set):
    """
    Build {model_key: tensor} for keys in model_keys.
    Fills *.weight/bias from *.base.* and merges LoRA deltas into base weight.
    Supports .A/.B, .lora_A/.lora_B, .lora.down/.lora.up with alpha scaling.
    """
    mapped = _remap_suffix_keys(sd, model_keys)

    # Fill from *.base.weight/bias if needed
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
        return float(r)  # default scale=r

    # .lora_A/.lora_B
    for a_key in [k for k in sd if k.endswith(".lora_A.weight")]:
        prefix = a_key[:-len(".lora_A.weight")]
        b_key  = prefix + ".lora_B.weight"
        base_w = prefix + ".weight"
        if b_key in sd and base_w in mapped:
            A, B = sd[a_key], sd[b_key]
            if getattr(A, "ndim", 0)==2 and getattr(B, "ndim", 0)==2:
                r = int(A.shape[0]); s = _alpha(prefix, r)/max(1,r)
                mapped[base_w] = mapped[base_w] + (B @ A)*s

    # .lora.down/.lora.up
    for d_key in [k for k in sd if k.endswith(".lora.down.weight")]:
        prefix = d_key[:-len(".lora.down.weight")]
        u_key  = prefix + ".lora.up.weight"
        base_w = prefix + ".weight"
        if u_key in sd and base_w in mapped:
            A, B = sd[d_key], sd[u_key]
            if getattr(A, "ndim", 0)==2 and getattr(B, "ndim", 0)==2:
                r = int(A.shape[0]); s = _alpha(prefix, r)/max(1,r)
                mapped[base_w] = mapped[base_w] + (B @ A)*s

    # .A/.B
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

def _align_flat_from_sd(sd: dict, named):
    targets = {n for n,_ in named}
    sd_map = _dense_map_with_lora(sd, targets)
    parts = []
    for name, p in named:
        t = sd_map.get(name)
        if t is None:
            parts.append(p.detach().reshape(-1).cpu())  # fallback: keep current slice
        else:
            parts.append(t.detach().reshape(-1).cpu())
    return torch.cat(parts) if parts else torch.empty(0)



# ---------- args-file parser with inline comments ----------
class CommentArgParser(argparse.ArgumentParser):
    def __init__(self, *a, **kw):
        kw.setdefault("fromfile_prefix_chars", "@")
        super().__init__(*a, **kw)
    def convert_arg_line_to_args(self, line: str):
        line = line.strip()
        if not line or line.startswith("#"): return []
        return [tok for tok in shlex.split(line, comments=True) if tok]

# ---------- small utils ----------
def _fmt_secs(s: float) -> str:
    s = max(0.0, float(s))
    m, s = divmod(int(round(s)), 60); h, m = divmod(m, 60)
    if h: return f"{h}h {m}m {s}s"
    if m: return f"{m}m {s}s"
    return f"{s}s"

def _safe_torch_load(path, map_location='cpu'):
    try:  return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError: return torch.load(path, map_location=map_location)

HEAD_KEYS = ['classifier', 'score', 'head', 'out_proj', 'lm_head', 'classification_head']
def _is_head_param(name: str) -> bool:
    ln = name.lower(); return any(k in ln for k in HEAD_KEYS)
def _is_lora_param(name: str) -> bool:
    ln = name.lower(); return ln.endswith('.a') or ln.endswith('.b') or ('lora' in ln)

def iter_param_groups(model, scope: str, include_frozen: bool = True):
    for name, p in model.named_parameters():
        if (not include_frozen) and (not p.requires_grad): continue
        if scope == 'lora_only' and not _is_lora_param(name): continue
        if scope == 'lora_and_head' and not (_is_lora_param(name) or _is_head_param(name)): continue
        yield name, p

def flatten_with_map(named_params):
    if not named_params: raise RuntimeError("No parameters selected.")
    flat, slices, off = [], [], 0
    for name, t in named_params:
        n = t.numel()
        flat.append(t.detach().reshape(-1))
        slices.append((name, off, off+n, tuple(t.shape)))
        off += n
    return torch.cat(flat), slices

def set_params_from_flat(model, vec, slices):
    pmap = dict(model.named_parameters())
    with torch.no_grad():
        for name, a, b, shape in slices:
            if name in pmap:
                pmap[name].copy_(vec[a:b].view(shape))

def load_factory(factory_path: str):
    mod, fn = factory_path.split(':')
    return getattr(importlib.import_module(mod), fn)

def batch_loss(model, batch, device):
    import torch.nn.functional as F

    def _first_present(d, keys):
        for k in keys:
            if k in d:
                return d[k]
        return None

    # Accept dicts *and* Mapping-like objects (e.g., HF BatchEncoding)
    if isinstance(batch, Mapping) or hasattr(batch, "keys"):
        xb = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        # common label field names across toolchains
        y = _first_present(xb, ("labels", "label", "label_ids", "y", "targets", "target"))

        # Try the standard HF-style call first
        try:
            out = model(**xb)
        except TypeError:
            # Fallback for vision or custom models
            x = _first_present(xb, ("input_ids", "pixel_values", "images", "x", "inputs"))
            if x is None or y is None:
                raise RuntimeError("Could not find inputs/labels in batch mapping.")
            out = model(x)

    # Classic (x, y) tuples/lists
    elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
        x, y = batch[0], batch[1]
        x = x.to(device) if hasattr(x, "to") else x
        y = y.to(device) if hasattr(y, "to") else y
        out = model(x)

    else:
        # Last-resort introspection to help debugging instead of a silent fail
        raise RuntimeError(f"Unsupported batch type: {type(batch)} (has keys={hasattr(batch, 'keys')})")

    # Pull logits out of HF outputs or raw tensors
    logits = (out.get("logits") if isinstance(out, dict)
              else (out.logits if hasattr(out, "logits") else out))

    # Ensure y exists (some models return it inside output)
    if "labels" in out if isinstance(out, dict) else False:
        y = out["labels"]
    if y is None:
        # Try again from the original batch if not found yet
        if isinstance(batch, Mapping) or hasattr(batch, "keys"):
            y = _first_present(batch, ("labels", "label", "label_ids", "y", "targets", "target"))
    if y is None:
        raise RuntimeError("No labels found (looked for labels/label/label_ids/y/targets/target).")

    y = y.to(device).long().view(-1)

    # Flatten time/seq dims if needed
    if hasattr(logits, "ndim") and logits.ndim > 2:
        logits = logits.view(-1, logits.shape[-1])

    # If logits already look like probabilities, use NLL on log-probs
    if (hasattr(logits, "dtype") and logits.dtype.is_floating_point
        and torch.isfinite(logits).all()
        and logits.min() >= 0.0 and logits.max() <= 1.0
        and torch.allclose(logits.float().sum(dim=-1),
                           torch.ones_like(logits[..., 0]), atol=1e-3, rtol=1e-3)):
        return F.nll_loss((logits.clamp_min(1e-12)).log(), y, reduction="mean")
    else:
        return F.cross_entropy(logits, y, reduction="mean")

# --------- PAPER normalization (per tensor) ----------
def paper_normalize(vec_flat: torch.Tensor, slices, model) -> torch.Tensor:
    """Scale each param-slice so ||d_k|| = ||W_k||; zero any 1-D tensors."""
    v = vec_flat.clone()
    pmap = dict(model.named_parameters())
    eps = 1e-10
    with torch.no_grad():
        for name, a, b, shape in slices:
            if len(shape) <= 1:
                v[a:b].zero_()
                continue
            seg = v[a:b]
            dn = float(seg.norm().clamp_min(eps))
            wn = float(pmap[name].data.norm().clamp_min(eps))
            v[a:b] = seg * (wn / dn)
    return v

def _load_dir_flat(path):
    d = _safe_torch_load(path, map_location='cpu')
    if 'v' not in d: raise ValueError(f"Direction file missing key 'v': {path}")
    if 'proj_ranges' not in d: d['proj_ranges'] = {'x':(-1.0,1.0),'y':(-1.0,1.0)}
    return d

# ---------- grid eval with ETA + row heartbeats ----------
def eval_grid(model, base_flat, v1, v2, slices, loader, num_batches, device, xs, ys,
              print_every=200, print_rows=True):
    model.eval()
    Z = torch.zeros(len(xs), len(ys))
    total, counter, avg_pt = len(xs)*len(ys), 0, None
    with torch.inference_mode():
        for i, x in enumerate(xs):
            row_t0 = time.time()
            if print_rows:
                if avg_pt is None:
                    print(f"[2D][ROW] starting row {i+1}/{len(xs)}")
                else:
                    done = i*len(ys); rem = total - done
                    print(f"[2D][ROW] starting row {i+1}/{len(xs)} | per-pt≈{avg_pt:.3f}s | ETA full ~{_fmt_secs(avg_pt*rem)}")
            for j, y in enumerate(ys):
                vec = base_flat + x*v1 + y*v2
                set_params_from_flat(model, vec, slices)
                t1 = time.time()
                tot, n = 0.0, 0
                for b_idx, batch in enumerate(loader):
                    tot += float(batch_loss(model, batch, device)); n += 1
                    if n >= num_batches: break
                Z[i,j] = tot / max(1,n)
                counter += 1
                dt = time.time() - t1
                avg_pt = dt if avg_pt is None else 0.9*avg_pt + 0.1*dt
                if counter == 5:
                    print(f"[2D][ETA] Initial ETA ~{_fmt_secs(avg_pt*(total-counter))} (points={total}, per-pt≈{avg_pt:.3f}s)")
                if (counter % max(1, print_every)) == 0:
                    rem = avg_pt*(total-counter)
                    print(f"[2D][PROG] {counter}/{total} ({counter/total*100:.1f}%) per-pt≈{avg_pt:.3f}s | ETA ~{_fmt_secs(rem)}")
            if print_rows:
                print(f"[2D][ROW] finished row {i+1}/{len(xs)} in {_fmt_secs(time.time()-row_t0)}")
    print(f"[2D] Grid evaluation complete (points={total}, per-pt≈{avg_pt:.3f}s).")
    return Z

def main():
    ap = CommentArgParser(description='2D loss landscape (paper-style normalization, val loss) with optional path overlay.')
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--factory', required=True)
    ap.add_argument('--dirs-subdir', default=None)
    ap.add_argument('--out-subdir', default=None)
    ap.add_argument('--dir-x', default=None)
    ap.add_argument('--dir-y', default=None)
    ap.add_argument('--random-plane', action='store_true')
    ap.add_argument('--seed', type=int, default=42)

    ap.add_argument('--scope', choices=['trainable','lora_only','lora_and_head'], default='trainable')
    ap.add_argument('--include-frozen', type=lambda s: s.lower()!='false', default=True)

    ap.add_argument('--grid', type=int, default=101)
    ap.add_argument('--num-batches', type=int, default=8)  # paper-ish
    ap.add_argument('--levels', type=int, default=20)
    ap.add_argument('--cmap', default='summer')
    ap.add_argument('--line-width', type=float, default=1.25)
    ap.add_argument('--label-fontsize', type=int, default=12)
    ap.add_argument('--equal-aspect', type=lambda s: s.lower()!='false', default=True)

    # Paper: use [-1,1] by default. Keep overrides for convenience.
    ap.add_argument('--x-min', type=float, default=-1.0)
    ap.add_argument('--x-max', type=float, default= 1.0)
    ap.add_argument('--y-min', type=float, default=-1.0)
    ap.add_argument('--y-max', type=float, default= 1.0)
    
    ap.add_argument('--center-by', choices=['auto','loss'], default='auto',
                    help='How to choose the base (0,0) weights. "auto" = factory default/best; "loss" = min val_loss epoch.')
    ap.add_argument('--center-epoch', type=int, default=None,
                    help='Explicit epoch to use as center (overrides --center-by).')
    ap.add_argument('--center-ckpt', type=str, default=None,
                    help='Explicit checkpoint path to use as center.')

    ap.add_argument('--plot-style', choices=['filled','lines'], default='lines')
    ap.add_argument('--upsample', type=int, default=1)

    ap.add_argument('--marker-size', type=float, default=28)
    ap.add_argument('--marker-face', default='auto')
    ap.add_argument('--marker-edge', default='auto')
    ap.add_argument('--path-color', default='auto')
    ap.add_argument('--path-width', type=float, default=2.0)
    ap.add_argument('--draw-arrows', type=lambda s: s.lower()!='false', default=True)
    ap.add_argument('--arrow-every', type=int, default=1)
    ap.add_argument('--arrow-head', type=float, default=0.6)

    ap.add_argument('--mark-origin', type=lambda s: s.lower()!='false', default=True)
    ap.add_argument('--out', default=None)

    ap.add_argument('--print-every', type=int, default=200)
    ap.add_argument('--print-rows', type=lambda s: s.lower()!='false', default=True)
    
    ap.add_argument('--boxify-ranges', type=lambda s: s.lower()!='false', default=False,
                    help='If true, set square range from projected path: [-r,r]^2 with r=max(x_max,y_max)*range_mult.')
    ap.add_argument('--range-mult', type=float, default=1.05,
                    help='Multiplier for r computed from path (default 1.05).')
    ap.add_argument('--x-center', type=float, default=0.0)
    ap.add_argument('--y-center', type=float, default=0.0)
    
    args = ap.parse_args()


    def _xp(p): return os.path.expanduser(os.path.expandvars(p)) if p else p
    args.run_dir = _xp(args.run_dir); args.dir_x = _xp(args.dir_x); args.dir_y = _xp(args.dir_y); args.out = _xp(args.out)

    print(f"[2D] Start: run-dir={args.run_dir} grid={args.grid}x{args.grid} num_batches={args.num_batches}")
    build = load_factory(args.factory)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[2D] Building eval model on device={device} ...")
    model, eval_loader = build(run_dir=args.run_dir, device=device)
    device = next(model.parameters()).device

    if args.scope == "lora_and_head":
        has_lora = any(_is_lora_param(n) for n,_ in model.named_parameters())
        if not has_lora:
            print("[WARN] --scope lora_and_head requested but no LoRA params detected; plotting head-only.")

    named = [(n,p) for n,p in iter_param_groups(model, args.scope, include_frozen=args.include_frozen)]
    if not named:
        named = [(n,p) for n,p in iter_param_groups(model, 'trainable', include_frozen=True)]
        print("[2D][WARN] No params matched scope; fallback to trainable(include_frozen=True).")
    base_flat, slices = flatten_with_map(named)
    # ---- choose center weights & override base_flat if requested ----
    center_epoch = args.center_epoch
    center_ckpt = _xp(args.center_ckpt) if args.center_ckpt else None

    if center_ckpt is None and center_epoch is None and args.center_by == 'loss':
        los = _best_epoch_by_loss(args.run_dir)
        if los is not None:
            center_epoch, _, center_ckpt = los

    if center_ckpt is None and center_epoch is not None:
        center_ckpt = _choose_ckpt_for_epoch(args.run_dir, center_epoch)

    if center_ckpt is None and args.center_by == 'auto':
        # try to honor PCA meta if directions came with it
        try:
            d_meta = _safe_torch_load(args.dir_x, map_location='cpu') if args.dir_x else {}
            if isinstance(d_meta, dict) and d_meta.get('best_ckpt'):
                # If was saved as a bare name, search in run_dir
                candidate = d_meta['best_ckpt']
                if os.path.isabs(candidate) and os.path.isfile(candidate):
                    center_ckpt = candidate
                else:
                    for p in _find_ckpts(args.run_dir):
                        if os.path.basename(p) == os.path.basename(candidate):
                            center_ckpt = p; break
        except Exception:
            pass

    if center_ckpt is not None and os.path.isfile(center_ckpt):
        try:
            sd = _load_state_dict_candidates(center_ckpt)
            vec = _align_flat_from_sd(sd, named).to(device)
            if vec.numel() == base_flat.numel():
                base_flat = vec
                print(f"[2D][CENTER] Using center checkpoint: {os.path.basename(center_ckpt)}")
            else:
                print(f"[2D][CENTER][WARN] Aligned vector size mismatch; keeping factory center.")
        except Exception as e:
            print(f"[2D][CENTER][WARN] Failed to align center ckpt: {e}")
    else:
        if args.center_ckpt or args.center_epoch or args.center_by == 'loss':
            print("[2D][CENTER][WARN] Could not resolve a center checkpoint; using factory-loaded weights.")
    print(f"[2D] Params: dims={base_flat.numel():,} tensors={len(slices)}")

    # Resolve directions
    if (not args.dir_x or not args.dir_y) and args.dirs_subdir:
        base = os.path.join(args.run_dir, args.dirs_subdir)
        cx, cy = os.path.join(base,'dirX.pth'), os.path.join(base,'dirY.pth')
        if os.path.isfile(cx) and os.path.isfile(cy):
            args.dir_x, args.dir_y = cx, cy
            print(f"[2D] Resolved directions from subdir: {base}")

    use_random = bool(args.random_plane or (not args.dir_x or not args.dir_y))
    print(f"[2D] Directions: {'RANDOM PLANE' if use_random else 'FROM FILES'}; scope={args.scope} include_frozen={args.include_frozen}")

    if use_random:
        g = torch.Generator(device=device).manual_seed(args.seed)
        v1_raw = torch.randn_like(base_flat, generator=g)
        v2_raw = torch.randn_like(base_flat, generator=g)
        xr = (-1.0, 1.0); yr = (-1.0, 1.0)
    else:
        d1 = _load_dir_flat(args.dir_x); d2 = _load_dir_flat(args.dir_y)
        v1_raw = d1['v'].to(device); v2_raw = d2['v'].to(device)
        xr = tuple(d1.get('proj_ranges',{}).get('x', (-1.0,1.0)))
        yr = tuple(d2.get('proj_ranges',{}).get('y', (-1.0,1.0)))
        names_shapes = [(n, tuple(int(x) for x in shape)) for n,_,_,shape in slices]
        cur_fp = hashlib.sha1(repr(names_shapes).encode()).hexdigest()[:16]
        meta_fp = d1.get('fingerprint')
        if meta_fp and meta_fp != cur_fp:
            print(f"[WARN] Direction fingerprint mismatch (dir vs current params): {meta_fp} != {cur_fp}")

    # PAPER normalization (per-tensor) before evaluating loss
    v1 = paper_normalize(v1_raw, slices, model)
    v2 = paper_normalize(v2_raw, slices, model)

    # Final ranges (paper default is [-1,1], but allow override via args and/or file meta)
    x_min = args.x_min if args.x_min is not None else xr[0]
    x_max = args.x_max if args.x_max is not None else xr[1]
    y_min = args.y_min if args.y_min is not None else yr[0]
    y_max = args.y_max if args.y_max is not None else yr[1]

    # Optional: boxify by gradient path so range = 1.05 * max extent, square and centered
    if args.boxify_ranges:
        # find path (same resolution logic as the overlay)
        path_pt = os.path.join(args.run_dir, 'landscape_pca', 'path_xy.pt')
        if not os.path.isfile(path_pt) and args.dir_x:
            cand = os.path.join(os.path.dirname(args.dir_x), 'path_xy.pt')
            if os.path.isfile(cand):
                path_pt = cand
        if os.path.isfile(path_pt):
            try:
                P = _safe_torch_load(path_pt, map_location='cpu')
                XY = P.get('XY', None)
                if isinstance(XY, torch.Tensor) and XY.ndim == 2 and XY.size(0) == 2:
                    px, py = XY[0].tolist(), XY[1].tolist()
                    cx, cy = float(args.x_center), float(args.y_center)
                    x_max_path = max((abs(p - cx) for p in px), default=0.0)
                    y_max_path = max((abs(p - cy) for p in py), default=0.0)
                    r = max(x_max_path, y_max_path) * float(args.range_mult)
                    if r > 0:
                        x_min, x_max = cx - r, cx + r
                        y_min, y_max = cy - r, cy + r
                        print(f"[2D][BOX] Using square range from path: r={r:.4f} center=({cx},{cy})")
                else:
                    print("[2D][BOX] path_xy.pt present but missing XY tensor; keeping defaults.")
            except Exception as e:
                print(f"[2D][BOX][WARN] Failed to compute box ranges from path: {e}")
        else:
            print("[2D][BOX] No path_xy.pt found; keeping defaults.")

    print(f"[2D][RANGE] X[{x_min}, {x_max}] Y[{y_min}, {y_max}]")


    xs = torch.linspace(x_min, x_max, args.grid, device=device)
    ys = torch.linspace(y_min, y_max, args.grid, device=device)

    if args.print_every <= 0:
        args.print_every = max(100, (len(xs)*len(ys))//20)
        print(f"[2D] print-every set to {args.print_every} (auto)")

    Z = eval_grid(model, base_flat, v1, v2, slices, eval_loader, args.num_batches, device,
                  xs, ys, print_every=args.print_every, print_rows=args.print_rows)

    # Optional upsample for smoother contours
    if args.upsample and args.upsample > 1:
        import torch.nn.functional as nnF
        Z = nnF.interpolate(Z.unsqueeze(0).unsqueeze(0), scale_factor=args.upsample,
                            mode='bicubic', align_corners=False).squeeze(0).squeeze(0)
        xs = torch.linspace(x_min, x_max, Z.size(0), device=Z.device)
        ys = torch.linspace(y_min, y_max, Z.size(1), device=Z.device)

    Xs, Ys = torch.meshgrid(xs.cpu(), ys.cpu(), indexing='ij')
    plt.figure(figsize=(8,8))
    if args.plot_style == 'filled':
        cf = plt.contourf(Xs.numpy(), Ys.numpy(), Z.cpu().numpy(), levels=args.levels, cmap=args.cmap)
        plt.colorbar(cf, label='loss')
    else:
        cs = plt.contour(Xs.numpy(), Ys.numpy(), Z.cpu().numpy(), levels=args.levels,
                         cmap=args.cmap, linewidths=args.line_width)
        plt.clabel(cs, inline=True, fontsize=args.label_fontsize)

    if args.equal_aspect:
        ax = plt.gca(); ax.set_aspect('equal', adjustable='box')
    ax = plt.gca(); ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)

    # overlay path if available
    path_pt = os.path.join(args.run_dir, 'landscape_pca', 'path_xy.pt')
    if not os.path.isfile(path_pt) and args.dir_x:
        cand = os.path.join(os.path.dirname(args.dir_x), 'path_xy.pt')
        if os.path.isfile(cand): path_pt = cand
    if os.path.isfile(path_pt):
        try:
            P = _safe_torch_load(path_pt, map_location='cpu')
            XY = P.get('XY', None)
            if isinstance(XY, torch.Tensor) and XY.ndim == 2 and XY.size(0) == 2:
                px, py = XY[0].tolist(), XY[1].tolist()
                path_color = 'white' if args.plot_style == 'filled' else 'tab:blue' if args.path_color=='auto' else args.path_color
                m_face = path_color if args.marker_face=='auto' else args.marker_face
                m_edge = 'black' if args.marker_edge=='auto' else args.marker_edge
                plt.plot(px, py, '-', color=path_color, linewidth=args.path_width, zorder=11)
                plt.scatter(px, py, s=args.marker_size, facecolors=m_face, edgecolors=m_edge, linewidths=0.8, zorder=12)
                if args.draw_arrows and len(px) > 1:
                    import numpy as np
                    px_np, py_np = np.array(px), np.array(py)
                    step = max(1, int(args.arrow_every))
                    base_ms = 10.0 + 2.0*float(args.path_width)
                    scale = max(0.5, min(2.0, float(args.arrow_head)))
                    _ms = base_ms*scale; _lw = max(0.8, 0.8*float(args.path_width))
                    for i in range(0, len(px)-1, step):
                        plt.annotate("", xy=(px_np[i+1], py_np[i+1]), xytext=(px_np[i], py_np[i]),
                                     arrowprops=dict(arrowstyle="-|>", color=path_color, lw=_lw, mutation_scale=_ms),
                                     zorder=13)
                print(f"[2D] Overlaid path with {len(px)} points.")
        except Exception as e:
            print(f"[2D][WARN] Failed to overlay path: {e}")

    if args.mark_origin:
        plt.plot([0],[0], marker='+', color='k', markersize=8, markeredgewidth=1.5, zorder=14)

    plt.xlabel('dirX'); plt.ylabel('dirY')
    subdir = args.out_subdir or args.dirs_subdir
    if args.out is None:
        fname = 'landscape_2d_with_path.png'
        out_path = os.path.join(args.run_dir, subdir, fname) if subdir else os.path.join(args.run_dir, fname)
    else:
        out_path = args.out; os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path, dpi=220)
    print(f"[2D] Saved contour to {out_path}.")

if __name__ == '__main__':
    main()
