# flyguy/fly.py -> flat f32 blob + Lua manifest. Rebuilds the in-play graph, asserts against the checkpoint.
import base64, json
import numpy as np, torch

def reach(start, a, b, n, hops):
    out = [np.zeros(n, bool)]
    out[0][start] = True
    for _ in range(hops):
        nxt = out[-1].copy(); nxt[b[out[-1][a]]] = True; out.append(nxt)
    return out

def csr(rows, n_rows):
    return np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=n_rows))]).astype(np.int64)

g = np.load("graph.npz")
n = len(g["body"])
pre, post = g["pre"].astype(np.int64), g["post"].astype(np.int64)
sens, dn = g["sensory"].astype(np.int64), g["descending"].astype(np.int64)
HOPS, FAN, SEED = 3, 16, 0
assert HOPS <= int(g["hops"])

ckpt = torch.load("model.pt", map_location="cpu", weights_only=False)
cfg, sd = ckpt["config"]["model"], ckpt["model"]
OBS_DIM = 137 + 4 * 34 + (24 if cfg.get("ball_pred") else 0) + 5 * (33 + (11 if cfg.get("opp_inputs") else 0))

fwd, bwd = reach(sens, pre, post, n, HOPS), reach(dn, post, pre, n, HOPS)
sets = [np.nonzero(fwd[k] & bwd[HOPS - k])[0] for k in range(HOPS + 1)]
sizes = [len(s) for s in sets]
pos = []
for s in sets:
    q = np.full(n, -1, np.int64); q[s] = np.arange(len(s)); pos.append(q)
is_sens = np.zeros(n, bool); is_sens[sens] = True

hop, used = {}, 0
for k in range(1, HOPS + 1):
    m = np.nonzero((pos[k - 1][pre] >= 0) & (pos[k][post] >= 0))[0]
    r, c = pos[k][post[m]], pos[k - 1][pre[m]]
    o = np.lexsort((c, r)); m, r, c = m[o], r[o], c[o]
    t = np.lexsort((r, c))
    hop[k] = dict(e=m, tcrow=csr(c[t], len(sets[k - 1])), tcol=r[t], tperm=t,
                  s=np.nonzero(is_sens[sets[k]])[0], d=pos[0][sets[k][is_sens[sets[k]]]])
    used += len(m)

dn_rows = pos[HOPS][dn[pos[HOPS][dn] >= 0]]
rng = np.random.default_rng(SEED)
n_s = sizes[0]
cols = np.stack([np.sort(rng.choice(OBS_DIM, size=FAN, replace=False)) for _ in range(n_s)]).reshape(-1)
in_crow = csr(np.repeat(np.arange(n_s), FAN), n_s)

thr = torch.nn.functional.softplus(torch.as_tensor(sd["actor.theta"], dtype=torch.float64)).numpy()
def A(k): return np.ascontiguousarray(sd[k].numpy(), dtype=np.float32)
W, gain, in_w, in_b = A("actor.w"), A("actor.gain"), A("actor.in_w"), A("actor.in_b")
outw, outb = A("actor.out.weight"), A("actor.out.bias")
assert in_w.shape == (n_s, FAN), f"in_w {in_w.shape}"
assert W.shape[0] == used, f"w {W.shape[0]} != used {used}"
assert outw.shape == (90, len(dn_rows)), f"out {outw.shape}"

blob, off = bytearray(), {}
def put(name, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    off[name] = [len(blob) // 4, arr.size]; blob.extend(arr.tobytes())

put("in_crow", in_crow); put("in_c", cols); put("in_w", in_w); put("in_b", in_b)
put("g0", gain[sets[0]]); put("t0", thr[sets[0]])
for k in range(1, HOPS + 1):
    h = hop[k]
    put(f"tcrow{k}", h["tcrow"]); put(f"tcol{k}", h["tcol"]); put(f"tw{k}", W[h["e"]][h["tperm"]])
    put(f"g{k}", gain[sets[k]]); put(f"t{k}", thr[sets[k]])
    put(f"s{k}", h["s"]); put(f"d{k}", h["d"]); put(f"s{k}_n", [len(h["s"])])
put("dn", dn_rows); put("ow", outw); put("ob", outb)
print(f"obs_dim={OBS_DIM} sizes={sizes} used={used} dn={len(dn_rows)} bytes={len(blob)}")

lines = ["-- generated, do not edit", "return {", f"  obs_dim = {OBS_DIM}, hops = {HOPS}, fan = {FAN}, n_actions = 90,",
         "  sizes = " + json.dumps(sizes), "  off = " + json.dumps(off),
         f'  data = "{base64.b64encode(bytes(blob)).decode()}",', "}"]
open("flyguy_weights.lua", "w").write("\n".join(lines) + "\n")
