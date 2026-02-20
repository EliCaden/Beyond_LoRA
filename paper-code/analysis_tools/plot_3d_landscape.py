# =============================
# plot_3d_landscape.py — PAPER MODE
# 3D loss surface with paper-style normalization & [-1,1] ranges.
# - Directions are raw SVD/PCA vectors (flat) saved by weights_pca.py
# - We paper-normalize against CURRENT model before evaluating loss.
# - Loss from VALIDATION loader.
# - Optional VTK export (Z=loss).
# =============================
import argparse, os, importlib, sys, time, shlex, hashlib
from collections.abc import Mapping
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
    
import json, re


import analysis_tools.compat_case

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colormaps as cmaps
import numpy as np

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


# ---------- args-file parser ----------
class CommentArgParser(argparse.ArgumentParser):
    def __init__(self, *a, **kw):
        kw.setdefault("fromfile_prefix_chars", "@")
        super().__init__(*a, **kw)
    def convert_arg_line_to_args(self, line: str):
        line = line.strip()
        if not line or line.startswith("#"): return []
        return [tok for tok in shlex.split(line, comments=True) if tok]

def fmt_secs(s: float) -> str:
    s = max(0.0, float(s))
    m, s = divmod(int(round(s)), 60); h, m = divmod(m, 60)
    if h: return f"{h}h {m}m {s}s"
    if m: return f"{m}m {s}s"
    return f"{s}s"

HEAD_KEYS = ['classifier','score','head','out_proj','lm_head','classification_head']
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

def load_factory(path: str):
    m,f = path.split(':')
    return getattr(importlib.import_module(m), f)

def _safe_torch_load(path, map_location='cpu'):
    try:  return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError: return torch.load(path, map_location=map_location)

def _load_dir_flat(path):
    d = _safe_torch_load(path, map_location='cpu')
    if 'v' not in d: raise ValueError(f"Direction file missing key 'v': {path}")
    if 'proj_ranges' not in d: d['proj_ranges'] = {'x':(-1.0,1.0),'y':(-1.0,1.0)}
    return d

def batch_loss(model, batch, device):
    import torch.nn.functional as F

    def _first_present(d, keys):
        for k in keys:
            if k in d:
                return d[k]
        return None

    # Accept Mapping-like (e.g., HF BatchEncoding) and tuples
    if isinstance(batch, Mapping) or hasattr(batch, "keys"):
        xb = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        y = _first_present(xb, ("labels", "label", "label_ids", "y", "targets", "target"))
        try:
            out = model(**xb)
        except TypeError:
            x = _first_present(xb, ("input_ids", "pixel_values", "images", "x", "inputs"))
            if x is None or y is None:
                raise RuntimeError("Could not find inputs/labels in batch mapping.")
            out = model(x)
    elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
        x, y = batch[0], batch[1]
        x = x.to(device) if hasattr(x, "to") else x
        y = y.to(device) if hasattr(y, "to") else y
        out = model(x)
    else:
        raise RuntimeError(f"Unsupported batch type: {type(batch)} (has keys={hasattr(batch, 'keys')})")

    logits = (out.get("logits") if isinstance(out, dict)
              else (out.logits if hasattr(out, "logits") else out))

    if isinstance(out, dict) and "labels" in out:
        y = out["labels"]
    if y is None and (isinstance(batch, Mapping) or hasattr(batch, "keys")):
        y = _first_present(batch, ("labels", "label", "label_ids", "y", "targets", "target"))
    if y is None:
        raise RuntimeError("No labels found (looked for labels/label/label_ids/y/targets/target).")

    y = y.to(device).long().view(-1)
    if hasattr(logits, "ndim") and logits.ndim > 2:
        logits = logits.view(-1, logits.shape[-1])

    if (hasattr(logits, "dtype") and logits.dtype.is_floating_point
        and torch.isfinite(logits).all()
        and logits.min() >= 0.0 and logits.max() <= 1.0
        and torch.allclose(logits.float().sum(dim=-1),
                           torch.ones_like(logits[..., 0]), atol=1e-3, rtol=1e-3)):
        return F.nll_loss((logits.clamp_min(1e-12)).log(), y, reduction="mean")
    else:
        return F.cross_entropy(logits, y, reduction="mean")

# PAPER normalization
def paper_normalize(vec_flat: torch.Tensor, slices, model, *, ignore_bias_bn: bool = True) -> torch.Tensor:
    """
    Filter-wise (per output channel / per row) normalization, matching Li et al.
    - For Conv: scale each out_channel independently.
    - For Linear: scale each out_feature row independently.
    - Optionally ignore bias/BN/1-D tensors by zeroing (default=True).
    """
    v = vec_flat.clone()
    pmap = dict(model.named_parameters())
    eps = 1e-10
    with torch.no_grad():
        for name, a, b, shape in slices:
            if name not in pmap:
                continue
            lname = name.lower()

            # Ignore 1-D params (bias, norm scales/shifts) if requested
            if ignore_bias_bn and (len(shape) <= 1):
                v[a:b].zero_()
                continue

            if len(shape) >= 2:
                seg = v[a:b].view(shape)           # direction slice for this param
                w   = pmap[name].data.view(shape)  # current weights at the center

                # Treat dim0 as "filters": Conv(out,in,k,k) or Linear(out,in)
                seg_flat = seg.view(shape[0], -1)
                w_flat   = w.view(shape[0], -1)

                dn = seg_flat.norm(dim=1).clamp_min(eps)   # direction norms per filter
                wn = w_flat.norm(dim=1).clamp_min(eps)     # weight norms per filter
                scale = (wn / dn).view(-1, *([1] * (seg.dim() - 1)))

                seg.mul_(scale)
                v[a:b] = seg.reshape(-1)
            else:
                # keep 1-D params if not ignoring (rare for this use case)
                pass
    return v


# evaluation
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
                    print(f"[3D][ROW] starting row {i+1}/{len(xs)}")
                else:
                    done = i*len(ys); rem = total - done
                    print(f"[3D][ROW] starting row {i+1}/{len(xs)} | per-pt≈{avg_pt:.3f}s | ETA full ~{fmt_secs(avg_pt*rem)}")
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
                    print(f"[3D][ETA] Initial full surface ETA: ~{fmt_secs(avg_pt*(total-counter))} (points={total}, per-pt≈{avg_pt:.3f}s)")
                if (counter % max(1, print_every)) == 0:
                    rem = avg_pt*(total-counter)
                    print(f"[3D][PROG] {counter}/{total} ({counter/total*100:.1f}%) per-pt≈{avg_pt:.3f}s | ETA full ~{fmt_secs(rem)}")
            if print_rows:
                print(f"[3D][ROW] finished row {i+1}/{len(xs)} in {fmt_secs(time.time()-row_t0)}")
    print(f"[3D] Grid evaluation complete (points={total}, per-pt≈{avg_pt:.3f}s).")
    return Z

def save_vtk_surface(path, xs, ys, Z):
    xs = np.asarray(xs); ys = np.asarray(ys); Z = np.asarray(Z)
    Ni, Nj = Z.shape
    with open(path, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\nLoss surface\nASCII\n")
        f.write("DATASET STRUCTURED_GRID\n")
        f.write(f"DIMENSIONS {Ni} {Nj} 1\n")
        f.write(f"POINTS {Ni*Nj} float\n")
        for i in range(Ni):
            for j in range(Nj):
                f.write(f"{xs[i]:.7g} {ys[j]:.7g} {Z[i,j]:.7g}\n")
        f.write(f"\nPOINT_DATA {Ni*Nj}\nSCALARS loss float 1\nLOOKUP_TABLE default\n")
        for i in range(Ni):
            for j in range(Nj):
                f.write(f"{Z[i,j]:.7g}\n")
    print(f"[3D] Saved VTK surface to {path}")

def main():
    ap = CommentArgParser(description='3D loss surface (paper-style normalization, val loss).')
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--factory', required=True)
    ap.add_argument('--dir-x', default=None)
    ap.add_argument('--dir-y', default=None)
    ap.add_argument('--random-plane', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--scope', choices=['trainable','lora_only','lora_and_head'], default='trainable')
    ap.add_argument('--include-frozen', type=lambda s: s.lower()!='false', default=True)

    ap.add_argument('--grid', type=int, default=49)
    ap.add_argument('--num-batches', type=int, default=8)  # paper-ish
    ap.add_argument('--dirs-subdir', default=None)
    ap.add_argument('--out-subdir', default=None)

    ap.add_argument('--cmap', default='coolwarm')
    ap.add_argument('--upsample', type=int, default=1)
    ap.add_argument('--z-exag', type=float, default=1.0)
    ap.add_argument('--shading', choices=['none','soft'], default='soft')
    ap.add_argument('--elev', type=float, default=35.0)
    ap.add_argument('--azim', type=float, default=-60.0)
    
    ap.add_argument('--center-by', choices=['auto','loss'], default='auto',
                    help='How to choose the base (0,0) weights. "auto" = factory default/best; "loss" = min val_loss epoch.')
    ap.add_argument('--center-epoch', type=int, default=None,
                    help='Explicit epoch to use as center (overrides --center-by).')
    ap.add_argument('--center-ckpt', type=str, default=None,
                    help='Explicit checkpoint path to use as center.')
    
    ap.add_argument('--boxify-ranges', type=lambda s: s.lower()!='false', default=False,
                    help='If true, set square range from projected path: [-r,r]^2 with r=max extent * range_mult.')
    ap.add_argument('--range-mult', type=float, default=1.05)
    ap.add_argument('--x-center', type=float, default=0.0)
    ap.add_argument('--y-center', type=float, default=0.0)



    # Paper ranges default
    ap.add_argument('--x-min', type=float, default=-1.0)
    ap.add_argument('--x-max', type=float, default= 1.0)
    ap.add_argument('--y-min', type=float, default=-1.0)
    ap.add_argument('--y-max', type=float, default= 1.0)

    ap.add_argument('--out', default=None)
    ap.add_argument('--save-vtk', default=None)

    ap.add_argument('--print-rows', type=lambda s: s.lower()!='false', default=True)
    ap.add_argument('--print-every', type=int, default=100)
    args = ap.parse_args()

    def _xp(p): return os.path.expanduser(os.path.expandvars(p)) if p else p
    args.run_dir = _xp(args.run_dir); args.dir_x = _xp(args.dir_x); args.dir_y = _xp(args.dir_y)
    args.out = _xp(args.out); args.save_vtk = _xp(args.save_vtk)

    print(f"[3D] Start: run-dir={args.run_dir} grid={args.grid}x{args.grid} num_batches={args.num_batches}")
    build = load_factory(args.factory)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[3D] Building eval model on device={device} ...")
    model, eval_loader = build(run_dir=args.run_dir, device=device)
    device = next(model.parameters()).device

    if args.scope == "lora_and_head":
        has_lora = any(_is_lora_param(n) for n,_ in model.named_parameters())
        if not has_lora:
            print("[WARN] --scope lora_and_head requested but no LoRA params detected; plotting head-only.")

    named = [(n,p) for n,p in iter_param_groups(model, args.scope, include_frozen=args.include_frozen)]
    if not named:
        named = [(n,p) for n,p in iter_param_groups(model, 'trainable', include_frozen=True)]
        print("[3D][WARN] No params matched scope; fallback to trainable(include_frozen=True).")
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
        try:
            # try PCA meta from dirX if available
            d_meta = _safe_torch_load(args.dir_x, map_location='cpu') if args.dir_x else {}
            if isinstance(d_meta, dict) and d_meta.get('best_ckpt'):
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
                print(f"[3D][CENTER] Using center checkpoint: {os.path.basename(center_ckpt)}")
            else:
                print(f"[3D][CENTER][WARN] Aligned vector size mismatch; keeping factory center.")
        except Exception as e:
            print(f"[3D][CENTER][WARN] Failed to align center ckpt: {e}")
    else:
        if args.center_ckpt or args.center_epoch or args.center_by == 'loss':
            print("[3D][CENTER][WARN] Could not resolve a center checkpoint; using factory-loaded weights.")

    print(f"[3D] Params: dims={base_flat.numel():,} tensors={len(slices)}")

    set_params_from_flat(model, base_flat, slices)

    # Resolve directions
    if (not args.dir_x or not args.dir_y) and args.dirs_subdir:
        base = os.path.join(args.run_dir, args.dirs_subdir)
        cx, cy = os.path.join(base,'dirX.pth'), os.path.join(base,'dirY.pth')
        if os.path.isfile(cx) and os.path.isfile(cy):
            args.dir_x, args.dir_y = cx, cy
            print(f"[3D] Resolved directions from subdir: {base}")
            
    use_random = bool(args.random_plane or (not args.dir_x or not args.dir_y))
    print(f"[3D] Directions: {'RANDOM PLANE' if use_random else 'FROM FILES'}; scope={args.scope} include_frozen={args.include_frozen}")

    if use_random:
        g = torch.Generator(device=device).manual_seed(args.seed)
        v1_raw = torch.randn(base_flat.shape, device=base_flat.device, generator=g)
        v2_raw = torch.randn(base_flat.shape, device=base_flat.device, generator=g)
        # Filter-normalize at the center
        v1 = paper_normalize(v1_raw, slices, model, ignore_bias_bn=True)
        v2 = paper_normalize(v2_raw, slices, model, ignore_bias_bn=True)
        xr = (-1.0, 1.0); yr = (-1.0, 1.0)
    else:
        d1 = _load_dir_flat(args.dir_x); d2 = _load_dir_flat(args.dir_y)
        v1_raw = d1['v'].to(device); v2_raw = d2['v'].to(device)
        # Keep your fingerprint check
        names_shapes = [(n, tuple(int(x) for x in shape)) for n,_,_,shape in slices]
        cur_fp = hashlib.sha1(repr(names_shapes).encode()).hexdigest()[:16]
        meta_fp = d1.get('fingerprint')
        if meta_fp and meta_fp != cur_fp:
            print(f"[WARN] Direction fingerprint mismatch (dir vs current params): {meta_fp} != {cur_fp}")

        # Filter-normalize PCA directions too (you asked to do this always)
        v1 = paper_normalize(v1_raw, slices, model, ignore_bias_bn=True)
        v2 = paper_normalize(v2_raw, slices, model, ignore_bias_bn=True)

        xr = tuple(d1.get('proj_ranges', {}).get('x', (-1.0, 1.0)))
        yr = tuple(d2.get('proj_ranges', {}).get('y', (-1.0, 1.0)))


    x_min = args.x_min if args.x_min is not None else xr[0]
    x_max = args.x_max if args.x_max is not None else xr[1]
    y_min = args.y_min if args.y_min is not None else yr[0]
    y_max = args.y_max if args.y_max is not None else yr[1]
    # Optional: boxify ranges around the projected training path (same logic as 2D)
    if args.boxify_ranges:
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
                        print(f"[3D][BOX] Using square range from path: r={r:.4f} center=({cx},{cy})")
                else:
                    print("[3D][BOX] path_xy.pt present but missing XY tensor; keeping defaults.")
            except Exception as e:
                print(f"[3D][BOX][WARN] Failed to compute box ranges from path: {e}")
        else:
            print("[3D][BOX] No path_xy.pt found; keeping defaults.")

    print(f"[3D][RANGE] Final ranges: X[{x_min}, {x_max}] Y[{y_min}, {y_max}]")


    xs = torch.linspace(x_min, x_max, args.grid, device=device)
    ys = torch.linspace(y_min, y_max, args.grid, device=device)
    if args.print_every <= 0:
        args.print_every = max(100, (len(xs)*len(ys))//20)
        print(f"[3D] print-every set to {args.print_every} (auto)")

    Z = eval_grid(model, base_flat, v1, v2, slices, eval_loader, args.num_batches, device,
                  xs, ys, print_every=args.print_every, print_rows=args.print_rows)

    # Center stats
    with torch.inference_mode():
        set_params_from_flat(model, base_flat, slices)
        tot, n = 0.0, 0
        for batch in eval_loader:
            tot += float(batch_loss(model, batch, device)); n += 1
            if n >= args.num_batches: break
        c_loss = tot / max(1,n)
    print(f"[3D] Center (0,0) loss={c_loss:.6f}; grid min={Z.min().item():.6f} max={Z.max().item():.6f}")

    # Upsample for smoother rendering/export
    if args.upsample and args.upsample > 1:
        import torch.nn.functional as nnF
        Zi = nnF.interpolate(Z.unsqueeze(0).unsqueeze(0), scale_factor=args.upsample,
                             mode='bicubic', align_corners=False).squeeze(0).squeeze(0)
        xs_i = torch.linspace(x_min, x_max, Zi.size(0))
        ys_i = torch.linspace(y_min, y_max, Zi.size(1))
    else:
        Zi, xs_i, ys_i = Z, xs.cpu(), ys.cpu()

    # default VTK path
    if args.save_vtk is None and (args.out_subdir or args.dirs_subdir):
        args.save_vtk = os.path.join(args.run_dir, (args.out_subdir or args.dirs_subdir), 'landscape_3d.vtk')
    if args.save_vtk:
        vtk_path = args.save_vtk if args.save_vtk.endswith('.vtk') else args.save_vtk + '.vtk'
        os.makedirs(os.path.dirname(vtk_path) or '.', exist_ok=True)
        save_vtk_surface(vtk_path, xs_i.numpy(), ys_i.numpy(), Zi.cpu().numpy())

    # Matplotlib 3D rendering
    Xs, Ys = torch.meshgrid(xs_i, ys_i, indexing='ij')
    Xs, Ys, Zi_np = Xs.cpu().numpy(), Ys.cpu().numpy(), Zi.cpu().numpy()
    cmap = cmaps.get_cmap(args.cmap)
    fig = plt.figure(figsize=(11,7)); ax = fig.add_subplot(111, projection='3d')
    if args.shading == 'soft':
        from matplotlib.colors import LightSource
        ls = LightSource(azdeg=315, altdeg=45)
        zmin, zmax = Zi_np.min(), Zi_np.max()
        zn = (Zi_np - zmin) / max(1e-12, (zmax - zmin))
        facecolors = ls.shade(zn, cmap=cmap, vert_exag=args.z_exag, blend_mode='soft')
        ax.plot_surface(Xs, Ys, Zi_np*args.z_exag, rstride=1, cstride=1, facecolors=facecolors,
                        linewidth=0, antialiased=True, shade=False)
    else:
        ax.plot_surface(Xs, Ys, Zi_np*args.z_exag, rstride=1, cstride=1, cmap=cmap,
                        linewidth=0, antialiased=True, shade=True)
    mappable = plt.cm.ScalarMappable(cmap=cmap); mappable.set_array(Zi_np)
    fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.05, label='loss')
    ax.set_xlabel('dirX'); ax.set_ylabel('dirY'); ax.set_zlabel('loss')
    ax.view_init(elev=args.elev, azim=args.azim); ax.set_box_aspect((1,1,1))

    subdir = args.out_subdir or args.dirs_subdir
    if args.out is None:
        fname = 'landscape_3d.png'
        out_path = os.path.join(args.run_dir, subdir, fname) if subdir else os.path.join(args.run_dir, fname)
    else:
        out_path = args.out; os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path, dpi=220)
    print(f"[3D] Saved surface to {out_path}.")

if __name__ == '__main__':
    main()
