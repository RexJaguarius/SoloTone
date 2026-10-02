#!/usr/bin/env python3
"""
NAM WaveNet inference engine (numpy — no torch/onnxruntime required).

A from-scratch reimplementation of the inference half of Neural Amp
Modeler's WaveNet architecture: loads a .nam capture file's JSON config
and flat weight array, and runs real-time streaming inference over
arbitrary-length audio blocks. Supports both the A1 and A2 NAM export
schemas (kernel_size vs kernel_sizes, gated vs gating_mode, a top-level
head vs a per-block head, etc.) and SlimmableContainer files (picks the
highest-quality WaveNet submodel). Does not support LSTM-architecture
files, FiLM conditioning, or a 'bottleneck' layer — see NamLoadError's
messages for exactly what a given file is missing.

Split out of solotone.py into its own module: this engine has zero
Tkinter/UI dependency (plain numpy in, numpy out), so it can be
imported, profiled, or unit-tested against real .nam files in isolation
from the rest of the app.

Public API: load_nam(path) -> (StreamingNAM, name); NamLoadError.
"""

import json
import os

import numpy as np

class NamLoadError(Exception): pass

def _act(name):
    """Resolve an activation name (string) or dict to a callable.
    A2 files store activations as dicts: {'type': 'LeakyReLU', 'negative_slope': 0.01}
    """
    if isinstance(name, dict):
        slope = float(name.get('negative_slope', 0.01))
        atype = name.get('type', 'LeakyReLU')
        if atype == 'LeakyReLU':
            return lambda x, s=slope: np.where(x > 0, x, s * x)
        name = atype  # fall through to string lookup

    acts = {
        'Tanh':     np.tanh,
        'ReLU':     lambda x: np.maximum(x, 0),
        'LeakyReLU':lambda x: np.where(x > 0, x, 0.01 * x),
        'Sigmoid':  lambda x: 1.0 / (1.0 + np.exp(-x)),
        'Identity': lambda x: x,
    }
    if name not in acts:
        raise NamLoadError(f"Unsupported activation '{name}'")
    return acts[name]

class _WR:
    """Weight reader — pulls values off the flat array in declaration order."""
    def __init__(self, w):
        self.w = np.asarray(w, dtype=np.float64); self.i = 0
    def take(self, n):
        chunk = self.w[self.i:self.i+n]
        if len(chunk) < n:
            raise NamLoadError("Ran out of weights — unsupported model variant")
        self.i += n; return chunk
    def left(self): return len(self.w) - self.i

class _Conv1d:
    def __init__(self, ci, co, k, d=1, bias=True):
        self.co, self.ci, self.k, self.d, self.has_b = co, ci, k, d, bias
        self.W = self.b = None
        self._state = None    # streaming history cache, (ci, (k-1)*d)
    def load(self, r):
        self.W = r.take(self.co*self.ci*self.k).reshape(self.co, self.ci, self.k)
        if self.has_b: self.b = r.take(self.co)
    def fwd(self, x):                          # x: (Ci, L)
        K, d = self.k, self.d
        ol = x.shape[1] - (K-1)*d
        if ol <= 0: raise NamLoadError("Input shorter than receptive field")
        out = np.zeros((self.co, ol))
        for k in range(K):
            out += self.W[:,:,k] @ x[:, k*d:k*d+ol]
        if self.b is not None: out += self.b[:,None]
        return out

    # ── incremental/streaming variant ──────────────────────────
    # A 1x1 conv (k=1) needs no history at all — every conv in this engine
    # except each layer's main dilated conv and the head conv is 1x1, so
    # this only actually caches anything for those two. That means true
    # per-block streaming can be pushed down to right here, instead of the
    # old approach of reprocessing the whole receptive field's worth of
    # raw audio through every layer on every single block (a ~50x
    # redundancy factor for a model with a few-thousand-sample receptive
    # field and a small streaming block size).
    def reset_state(self):
        hist_len = (self.k - 1) * self.d
        self._state = np.zeros((self.ci, hist_len))

    def fwd_stream(self, x_new):
        """x_new: (Ci, n_new) — only the NEW samples since the last call.
        Returns (Co, n_new). Maintains this conv's own small history cache
        so the result is identical to running fwd() on the full history
        each time, without redoing any of that prior work."""
        if self._state is None:
            self.reset_state()
        hist_len = self._state.shape[1]
        xin = np.concatenate([self._state, x_new], axis=1) if hist_len else x_new
        out = self.fwd(xin)
        if hist_len:
            self._state = xin[:, -hist_len:]
        return out

class _WNLayer:
    def __init__(self, cond_sz, ch, k, d, act_name, gated, act_fn=None):
        self.ch, self.gated = ch, gated
        mid = 2*ch if gated else ch
        self.conv  = _Conv1d(ch, mid, k, d)
        self.mix   = _Conv1d(cond_sz, mid, 1, bias=False)
        # act_fn takes priority over act_name (used by A2 per-layer activations)
        self.act   = act_fn if act_fn is not None else _act(act_name)
        self.proj  = _Conv1d(ch, ch, 1)
    def load(self, r):
        self.conv.load(r); self.mix.load(r); self.proj.load(r)
    def fwd(self, x, c, ol):
        zc = self.conv.fwd(x)
        zm = self.mix.fwd(c)[:, -zc.shape[1]:]
        z  = zc + zm
        if self.gated:
            post = self.act(z[:self.ch]) * (1/(1+np.exp(-z[self.ch:])))
        else:
            post = self.act(z)
        skip = post[:, -ol:]
        res  = x[:, -post.shape[1]:] + self.proj.fwd(post)
        return res, skip

    def reset_state(self):
        self.conv.reset_state(); self.mix.reset_state(); self.proj.reset_state()

    def fwd_stream(self, x_new, c_new):
        """x_new: (ch, n_new) new residual-stream samples from the previous
        layer (or rechan, for the first layer). c_new: (cond_sz, n_new) new
        raw conditioning samples — mix and proj are 1x1 so need no history;
        conv carries its own dilation history via _Conv1d.fwd_stream, so
        every output here is already exactly n_new samples — no truncation
        needed, unlike the reprocess-the-whole-window fwd() above."""
        zc = self.conv.fwd_stream(x_new)
        zm = self.mix.fwd_stream(c_new)
        z  = zc + zm
        if self.gated:
            post = self.act(z[:self.ch]) * (1/(1+np.exp(-z[self.ch:])))
        else:
            post = self.act(z)
        skip = post
        res  = x_new + self.proj.fwd_stream(post)
        return res, skip

class _WNBlock:
    """
    Handles both A1 and A2 layer-array schemas.

    A1 schema keys: input_size, condition_size, head_size, channels,
                    kernel_size (singular, int), dilations, activation,
                    gated, head_bias
    A2 schema keys: input_size, condition_size, channels, bottleneck (opt),
                    kernel_sizes (plural, list), dilations (list-of-ints OR
                    list-of-lists for restarting), activation (string or list),
                    head: {out_channels, kernel_size, bias}
                    (no top-level head_size or head_bias)
    """
    def __init__(self, cfg):
        self.in_sz   = cfg['input_size']
        self.cond_sz = cfg['condition_size']
        self.ch      = cfg['channels']

        # Kernel sizes: A1 uses 'kernel_size' (int), A2 uses 'kernel_sizes' (list)
        if 'kernel_sizes' in cfg:
            kss = list(cfg['kernel_sizes'])
        elif 'kernel_size' in cfg:
            ks = cfg['kernel_size']
            kss = ks if isinstance(ks, list) else None  # resolved after dilations
        else:
            kss = None  # default to 3 per layer

        # Dilations: A2 may use list-of-lists (restarting) or flat list
        raw_dil = cfg['dilations']
        if raw_dil and isinstance(raw_dil[0], list):
            # restarting dilations — flatten into a single sequence
            dilations = [d for group in raw_dil for d in group]
        else:
            dilations = list(raw_dil)

        if kss is None:
            kss = [3] * len(dilations)
        elif len(kss) != len(dilations):
            # A2 sometimes stores kernel_sizes once per restart group; tile it
            repeats = len(dilations) // len(kss)
            kss = kss * repeats + kss[:len(dilations) % len(kss)]

        # Activation: A2 may give a list (per-layer) or a string
        act_raw = cfg.get('activation', 'Tanh')
        if isinstance(act_raw, list):
            self.acts = [_act(a) for a in act_raw]
        else:
            self.acts = [_act(act_raw)] * len(dilations)

        # A1 uses 'gated' (bool). A2 uses 'gating_mode' (list of strings).
        # Default to False (ungated) when neither key is present — A2 standard.
        if 'gated' in cfg:
            self.gated = bool(cfg['gated'])
        elif 'gating_mode' in cfg:
            # gating_mode is a per-layer list; if any layer is not 'none', treat as gated
            gm = cfg['gating_mode']
            self.gated = isinstance(gm, list) and any(g != 'none' for g in gm)
        else:
            self.gated = False  # safe default for A2

        # Head: A1 has head_size+head_bias at block level;
        #       A2 has a 'head' sub-dict with out_channels+kernel_size+bias
        a2_head = cfg.get('head')
        if a2_head and isinstance(a2_head, dict):
            head_out  = int(a2_head.get('out_channels', 1))
            head_k    = int(a2_head.get('kernel_size', 1))
            head_bias = bool(a2_head.get('bias', True))
            self.hrec = _Conv1d(self.ch, head_out, head_k, bias=head_bias)
            self.head_sz = head_out
        else:
            # A1 path
            self.head_sz = int(cfg.get('head_size', 1))
            head_b       = bool(cfg.get('head_bias', True))
            self.hrec    = _Conv1d(self.ch, self.head_sz, 1, bias=head_b)

        self.rechan  = _Conv1d(self.in_sz, self.ch, 1, bias=False)
        # Build layers, passing per-layer activation
        self.layers  = [_WNLayer(self.cond_sz, self.ch, k, d,
                                  None, self.gated, act_fn=self.acts[i])
                        for i, (k, d) in enumerate(zip(kss, dilations))]
        self.kss     = kss
        # Receptive field of the DILATION STACK only.
        # The head conv kernel is handled separately in StreamingNAM.
        self.rf      = 1 + sum((k-1)*d for k,d in zip(kss, dilations))
        self.head_rf = self.hrec.k  # extra samples the head conv needs

    def load(self, r):
        self.rechan.load(r)
        for l in self.layers: l.load(r)
        self.hrec.load(r)

    def fwd(self, x, c, hi):
        ol = x.shape[1] - (self.rf-1)
        y  = self.rechan.fwd(x)
        for layer in self.layers:
            y, skip = layer.fwd(y, c, ol)
            hi = skip if hi is None else hi[:,-ol:] + skip
        return self.hrec.fwd(hi), y

    def reset_state(self):
        self.rechan.reset_state()
        for l in self.layers: l.reset_state()
        self.hrec.reset_state()

    def fwd_stream(self, x_new, c_new, hi_new):
        """x_new/c_new: (in_sz, n_new) new raw samples. hi_new: accumulated
        skip total carried in from a previous block, or None. Every value
        here is exactly n_new samples long — no ol/truncation bookkeeping
        needed, since each layer's own _Conv1d.fwd_stream already produces
        exactly n_new outputs from its own cached history."""
        y = self.rechan.fwd_stream(x_new)
        for layer in self.layers:
            y, skip = layer.fwd_stream(y, c_new)
            hi_new = skip if hi_new is None else hi_new + skip
        return self.hrec.fwd_stream(hi_new), y

class _WNHead:
    def __init__(self, cfg, in_ch):
        self.ch  = cfg['channels']
        self.act = _act(cfg['activation'])
        self.nl  = cfg['num_layers']
        self.oc  = cfg['out_channels']
        self.convs = []
        ci = in_ch
        for i in range(self.nl):
            co = self.ch if i < self.nl-1 else self.oc
            self.convs.append(_Conv1d(ci, co, 1)); ci = co
    def load(self, r):
        for c in self.convs: c.load(r)
    def fwd(self, x):
        y = x
        for c in self.convs: y = self.act(y); y = c.fwd(y)
        return y

    def reset_state(self):
        for c in self.convs: c.reset_state()

    def fwd_stream(self, x_new):
        # All convs here are 1x1 (see __init__) — no history needed, this
        # is just fwd() under a name that matches the rest of the
        # streaming call chain.
        return self.fwd(x_new)

class WaveNetNAM:
    def __init__(self, cfg, weights):
        try:
            self.blocks = [_WNBlock(bc) for bc in cfg['layers']]
        except KeyError as e:
            raise NamLoadError(
                f"Could not parse layer config — missing key {e}. "
                f"This may be an unsupported NAM variant."
            )
        # Top-level head: A1 may have one, A2 always has null/None
        hcfg = cfg.get('head')
        # A2 stores head inside each layer block — hcfg at top level will be None
        self.head = _WNHead(hcfg, self.blocks[-1].head_sz)                     if (hcfg and isinstance(hcfg, dict) and 'num_layers' in hcfg)                     else None
        self.hscale = float(cfg.get('head_scale', 1.0))
        # Total rf = cascade of dilation stacks + the final block's head conv
        self.rf      = 1 + sum(b.rf-1 for b in self.blocks)
        self.head_rf = self.blocks[-1].head_rf  # extra context for head conv
        r = _WR(weights)
        try:
            for b in self.blocks: b.load(r)
            if self.head: self.head.load(r)
        except NamLoadError as e:
            raise NamLoadError(
                f"Weight loading failed: {e}. "
                f"The file may use a NAM feature (FiLM conditioning, "
                f"bottleneck, etc.) not yet supported."
            )
        left = r.left()
        if left == 1: self.hscale = float(r.take(1)[0])
        elif left != 0:
            raise NamLoadError(
                f"{left} unused weights after loading. "
                f"This model may use FiLM conditioning or another "
                f"unsupported feature."
            )
    def fwd_stack(self, x1d):
        """Run the dilation stacks only. Returns the head-rechannel output
        (still in 2-D shape internally, returned as 1-D for the streaming buffer)."""
        x = np.asarray(x1d, np.float64)[None,:]
        y, hi = x, None
        for b in self.blocks:
            hi, y = b.fwd(y, x, hi)
        # hscale applies to the model's output regardless of A1 vs A2 — it
        # was previously only applied on the A1 branch below. For A2 files
        # (self.head is None, which every real-world .nam file loaded here
        # has been so far), the output came back completely unscaled: with
        # a head_scale of ~0.014 seen in the wild, that's a ~70x gain
        # excess baked directly into every sample, on top of whatever the
        # IR convolution added — this was the actual root cause of the
        # "loud, high-pitched clipping/screeching" bug, not an unmodeled
        # architecture feature.
        return (self.hscale * hi)[0]

    def fwd_head(self, stack_out_1d):
        """Run the top-level head (A1 only) or identity (A2).
        For A2 the head conv is inside the block and already ran in fwd_stack."""
        if self.head is None:
            # A2: stack_out already IS the final output
            return stack_out_1d
        # A1: run the top-level _WNHead
        hi = np.asarray(stack_out_1d, np.float64)[None,:]
        return self.head.fwd(hi)[0]

    def reset_state(self):
        for b in self.blocks: b.reset_state()
        if self.head: self.head.reset_state()

    def fwd_stack_stream(self, x1d_new):
        """Incremental counterpart to fwd_stack(): x1d_new is only the NEW
        raw samples for this call (length = the caller's block size, not
        the whole receptive field). Every conv down the call chain caches
        its own small dilation history (see _Conv1d.fwd_stream), so this
        produces the same result as fwd_stack() on the full history each
        time, but does none of the redundant recomputation — the model's
        conv/mix/proj/hrec calls have never resized here, only how much of
        the signal they're asked to reprocess each call."""
        x = np.asarray(x1d_new, np.float64)[None, :]
        y, hi = x, None
        for b in self.blocks:
            hi, y = b.fwd_stream(y, x, hi)
        return (self.hscale * hi)[0]

    def fwd_head_stream(self, stack_out_1d):
        # Same as fwd_head(): the top-level head (A1 only) is 1x1-only, so
        # there's no history to manage — it's just fwd_head() again.
        return self.fwd_head(stack_out_1d)

    def fwd(self, x1d):
        """Legacy single-call forward (used for testing). Not called in streaming."""
        stack = self.fwd_stack(x1d)
        return self.fwd_head(stack)

class StreamingNAM:
    """Stateful streaming wrapper for real-time block-by-block inference.

    Previously this reprocessed the model's entire receptive field (the
    last rf-1 raw samples, prepended to every new block) through all 23+
    layers on every single call — for a model with a several-thousand-
    sample receptive field and a small streaming block size, that's a
    ~50x redundancy factor (measured: ~6459 samples reprocessed to
    produce 128 new ones). Every conv in the call chain now caches its
    own small dilation history instead (_Conv1d.fwd_stream), so each
    layer only ever computes the new samples — there's no top-level
    history buffer to maintain here anymore.
    """
    def __init__(self, model):
        self.m  = model
        self.rf = model.rf   # kept for the "Receptive field: N ms" UI label
        self.m.reset_state()

    def reset(self):
        self.m.reset_state()

    def process(self, block):
        block     = np.asarray(block, np.float64)
        stack_out = self.m.fwd_stack_stream(block)
        return self.m.fwd_head_stream(stack_out).astype(np.float32)

def load_nam(path):
    with open(path) as f: data = json.load(f)
    arch = data.get('architecture','')

    # SlimmableContainer wraps multiple quality levels (lite/full) of a WaveNet.
    # Pick the highest-quality (largest max_value) submodel that is a WaveNet.
    if arch == 'SlimmableContainer':
        cfg = data.get('config', {})
        submodels = cfg.get('submodels', [])
        # Sort by max_value descending so we try the best model first
        submodels_sorted = sorted(
            submodels,
            key=lambda s: float(s.get('max_value', 0)),
            reverse=True
        )
        inner = None
        for sub in submodels_sorted:
            m = sub.get('model', {})
            if isinstance(m, dict) and m.get('architecture') == 'WaveNet':
                inner = m
                break
        if inner is None:
            raise NamLoadError(
                "SlimmableContainer file found but none of its submodels use "
                "the WaveNet architecture — cannot load this file."
            )
        data = {
            'architecture': 'WaveNet',
            'config':       inner.get('config', {}),
            'weights':      inner.get('weights', []),
            'metadata':     data.get('metadata', inner.get('metadata', {})),
        }
        arch = 'WaveNet'

    if arch != 'WaveNet':
        raise NamLoadError(
            "Only WaveNet-architecture .nam files are supported. "
            f"This file declares architecture: '{arch}'"
        )
    try:
        model = WaveNetNAM(data['config'], data['weights'])
    except NamLoadError:
        raise
    except Exception as e:
        raise NamLoadError(f"Unexpected error loading NAM file: {type(e).__name__}: {e}")
    meta  = data.get('metadata') or {}
    name  = meta.get('name') or os.path.splitext(os.path.basename(path))[0]
    return StreamingNAM(model), name
