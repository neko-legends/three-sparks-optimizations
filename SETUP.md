# SETUP — GLM-5.2 across 3× DGX Spark (RoCE ring)

> **Read this first if you are an AI agent.** This guide is phased and each phase
> has a **verify** block. Run the diagnostic checklist in §1 before anything
> else — it tells you which phase to start from. Do not skip §4 (the NCCL patch);
> the deployment hangs silently without it (a ~2-minute `ibv_modify_qp` timeout
> per queue pair, not a crash).

This guide reproduces a working, fast GLM-5.2 deployment across three DGX Spark
nodes (GB10 / Grace Blackwell, aarch64) wired in a **point-to-point RoCE ring**,
served with vLLM at tensor-parallel=3, with the model's full 8-expert quality
and 380k-token context.

It is layered on two existing open projects — nothing here replaces them:

- **Phase 1** is [NVIDIA's *Connect Three DGX Spark in a Ring Topology* playbook](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-three-sparks/README.md).
  It produces a working RoCE ring with IPs assigned and NCCL bandwidth-verified.
- **Phases 2–7** run [MiaAI-Lab's GLM-5.2 deployment](https://github.com/MiaAI-Lab/GLM-5.2-NVFP4-AQLM-Triple-DGX-Sparks)
  *on top of* that ring — with the patches in this repo applied so it actually
  completes. MiaAI-Lab's stock `./start.sh serve` hangs at NCCL init on the ring;
  this repo's §4 patch is the fix.

---

## 0. Prerequisites

- **3× DGX Spark** (GB10 SoC, sm_121, aarch64, 128 GB unified memory each), same
  OS/firmware on all three.
- **3× QSFP cables** for the ring (NVIDIA's [recommended cable](https://marketplace.nvidia.com/en-us/enterprise/personal-ai-supercomputers/qsfp-cable-0-4m-for-dgx-spark/) or equivalent).
- The same username on all three nodes. Root/sudo on all three.
- **CUDA 13.0** toolkit on each node (`nvcc` on PATH) — needed once, to rebuild NCCL.
- Docker on all three nodes. ~300 GB free on each for weights.
- An internet connection for the control plane (see Phase 2).

### Software this guide deploys

| Component | Source |
|---|---|
| Physical ring + netplan + NCCL sanity | NVIDIA `dgx-spark-playbooks` (Phase 1) |
| Control-plane networking | Tailscale MagicDNS (Phase 2) |
| GLM-5.2 weights (272 GB, NVFP4+AQLM) | MiaAI-Lab repo / HF (Phase 5) |
| vLLM (GLM-5.2 fork) + container image | `ghcr.io/miaai-lab/glm-5.2-nvfp4-triple-dgx-sparks:latest` |
| NCCL | NVIDIA NCCL contrib `fab1850` **+ this repo's §4 patch**, rebuilt locally |

---

## 1. Diagnostic checklist (run this first)

Before starting, figure out which phase you're already at. On each node run:

```bash
# (a) Are the CX-7 ports up with IPs?
ip -br addr show | grep -E "enp1s0f0np0|enp1s0f1np1"
# (b) Can each node ping its two neighbors over their shared subnets?
ping -c2 -I enp1s0f0np0 <neighbor-port0-ip>
ping -c2 -I enp1s0f1np1 <neighbor-port1-ip>
# (c) Is there a control-plane network resolving all three hostnames?
getent hosts node1 node2 node3   # or whatever you name them
# (d) Is passwordless SSH set up head→workers?
ssh node2 hostname && ssh node3 hostname
```

Interpret:

- **(a) fails** → start at **Phase 1** (the ring isn't wired/configured).
- **(a) passes but (b) fails** → Phase 1 netplan is wrong; recheck IP assignment.
- **(a)+(b) pass, (c) fails** → start at **Phase 2** (control plane).
- **(a)+(b)+(c) pass, no model yet** → start at **Phase 5** (skip to the model),
  but you **still need Phase 4's NCCL patch** — do not skip it.
- **Everything passes and the model is downloaded** → start at **Phase 6** (serve),
  after confirming Phase 4's patch is applied.

---

## Phase 1 — Build the RoCE ring (NVIDIA's playbook)

Follow NVIDIA's
[*Connect Three DGX Spark in a Ring Topology*](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-three-sparks/README.md)
playbook end to end. In summary it has you:

1. Cable the three Sparks in a ring: node1 port0↔node2 port1, node2 port0↔node3
   port1, node3 port0↔node1 port1. (Port 0 is the CX-7 port nearest the Ethernet
   jack; port 1 is the one further away.)
2. Assign each port a `/24` on its point-to-point link via netplan.
3. Set up passwordless SSH between nodes.
4. Run the NCCL bandwidth test to validate the ring.

The resulting topology is **three point-to-point `/24` links**, one per neighbor
pair — *not* a switched subnet:

```
              port0 (192.168.2.1)            port0 (192.168.2.2)
   node1 ─────────────────────────────────────── node2
     │ port1 (192.168.0.1)                          │ port1 (192.168.4.2)
     │                                              │
     └──────────── 192.168.0.x ────────────────────┘
                 node3: port1 (192.168.0.2)
                        port0 (192.168.4.1)  ←── node2↔node3 direct link
```

**This point-to-point structure is the entire reason Phase 4's patch is
required** — internalize it. Each node has two RoCE NICs but each *peer* is
reachable on only one subnet; node1 has no route to `192.168.4.0/24` at all.

Example netplan on node1 (repeat with the right addresses on node2/node3):

```bash
sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      dhcp4: false
      addresses: [ "192.168.2.1/24" ]
    enp1s0f1np1:
      dhcp4: false
      addresses: [ "192.168.0.1/24" ]
EOF
sudo chmod 600 /etc/netplan/40-cx7.yaml && sudo netplan apply
```

**Verify Phase 1:** `ping -I enp1s0f0np0 <peer-port-ip>` succeeds for every
neighbor pair, and NVIDIA's NCCL bandwidth test runs clean across all three nodes.

---

## Phase 2 — Control plane (Tailscale)

The RoCE ring is your **data plane**. You also need a **control plane** that
carries Ray/Gloo/SSH bootstrap traffic and resolves the three nodes by name. Any
flat L3 network works. **Tailscale MagicDNS is recommended** because DGX Sparks
sit on DHCP/WiFi where LAN IPs drift — MagicDNS gives each node a stable
hostname regardless of what IP the WiFi hands it.

On each node:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up                       # authenticate in the browser
tailscale status                        # confirm all 3 nodes appear
```

Each node gets a stable `100.x.y.z` tailnet IP and a MagicDNS name. Set them
consistently — e.g. name the nodes `node1`, `node2`, `node3` in the Tailscale
admin console (or accept the machine names). Confirm resolution:

```bash
getent hosts node1 node2 node3          # all three resolve to 100.x.y.z
ping -c2 node2 && ping -c2 node3        # reachable over the tailnet
```

The Tailscale interface is usually `tailscale0`. That's what you'll set
`NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` to in Phase 7 — the TCP bootstrap
for NCCL/Gloo rides the tailnet, while the actual tensor data rides RoCE.

> **Why a separate control plane?** NCCL's RoCE transport handles the bulk data,
> but the *initial* connection setup (socket bootstrap, GID exchange) uses plain
> TCP. That bootstrap needs all three nodes mutually reachable by a stable
> address. Doing it over the RoCE subnets is fragile (each NIC only sees one
> peer), and doing it over flaky DHCP WiFi makes startup nondeterministic.
> Tailscale sidesteps both.

**Verify Phase 2:** all three nodes resolve each other's MagicDNS names and ping
over `tailscale0`, and passwordless SSH works head→worker over those names.

---

## Phase 3 — Verify RoCE GID tables

On each node, confirm each CX-7 port's RoCEv2 IPv4 GID is at **GID index 3**:

```bash
show_gids | grep -E "enp1s0f0np0|enp1s0f1np1"
```

Expected (one line per port):

```
rocep1s0f0  1  3  0000:0000:0000:0000:0000:ffff:c0a8:0201  192.168.2.1  v2  enp1s0f0np0
```

The IPv4-mapped RoCEv2 GID **must be at index 3 on every port on every node** —
that's what `NCCL_IB_GID_INDEX=3` (set in Phase 7) relies on. If your layout
differs, set `NCCL_IB_GID_INDEX` to whatever index the `v2` / IPv4 GID occupies,
and it must match on all nodes.

---

## Phase 4 — The NCCL patch (REQUIRED — do not skip)

### Why it's required

On this ring, NCCL's graph layer assigns channels to NICs round-robin per peer,
so for a given peer half the channels get a NIC on a subnet the peer can't
reach. `ibv_modify_qp` then hangs ~2 minutes and times out:

```
NCCL WARN Call to ibv_modify_qp failed with 110 Connection timed out,
  on dev rocep1s0f0:1, local GID ::ffff:192.168.2.1, remote GID ::ffff:192.168.4.1
```

NVIDIA's NCCL contrib tree (commit `fab1850`, *"Support 3 DGX Sparks in a ring
topology via CX-7"*) has a `NCCL_IB_SUBNET_AWARE_ROUTING` feature *intended* to
fix this — the connector is supposed to pick a mutually-reachable NIC. It does
not work out of the box for a subtle reason: the connector can only choose among
GIDs the **listener** advertised, and `ncclIbListen` only embedded its *own
single device's* GID. When the graph assigned a listener to a NIC its peer can't
reach, the peer had no reachable GID to match, so the bad pairing was used.

### The fix

Patch `ncclIbListen` to embed GIDs from **all** Ethernet PFs (not just the
listening device). On each node, in the NCCL contrib source tree:

```bash
cd ~/nccl_spark_cluster        # the fab1850 contrib tree
cp src/transport/net_ib/connect.cc src/transport/net_ib/connect.cc.bak
```

Apply this diff to `src/transport/net_ib/connect.cc` (inside `ncclIbListen`,
around line 509):

```diff
-  // Embed GIDs of all PFs in the handle so the connector can find a local NIC
-  // on the same subnet as any of our ports.
-  if (ncclParamIbSubnetAwareRouting() && dev < ncclNMergedIbDevs) {
-    struct ncclIbMergedDev* mDev = ncclIbMergedDevs + dev;
+  // Embed GIDs of ALL our Ethernet PFs in the handle so the connector can
+  // find a local NIC on the same subnet as any of our ports. In a multi-subnet
+  // RoCE point-to-point ring (e.g. 3 DGX Sparks), a given listener device may
+  // be on a subnet the connector cannot reach; advertising every PF lets the
+  // connector pick a mutually-reachable subnet via ncclIbFindDevBySubnet().
+  if (ncclParamIbSubnetAwareRouting()) {
     int gidSlot = 0;
-    for (int i = 0; i < mDev->vProps.ndevs && gidSlot < 2; i++) {
-      int ibDevN = mDev->vProps.devs[i];
+    for (int ibDevN = 0; ibDevN < ncclNIbDevs && gidSlot < 2; ibDevN++) {
       ncclIbDev* ibDev = ncclIbDevs + ibDevN;
       if (ibDev->portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) continue;
       int gidIndex;
       NCCLCHECKGOTO(ncclIbGetGidIndex(ibDev->context, ibDev->portNum, &ibDev->portAttr, &gidIndex), ret, fail);
       NCCLCHECKGOTO(wrap_ibv_query_gid(ibDev->context, ibDev->portNum, gidIndex, &handle->listenGids[gidSlot]), ret, fail);
+      INFO(NCCL_NET, "NET/IB: Subnet-aware listen: dev %d (%s) GID[%d] embedded slot %d", dev, ibDev->devName, gidIndex, gidSlot);
       gidSlot++;
     }
   }
```

Rebuild (CUDA 13 on PATH):

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
make -j"$(nproc)" src.build
```

This produces `build/lib/libnccl.so.2.29.7`. Copy it to **all three nodes** at
the same path and create the symlinks on each:

```bash
cd ~/nccl_spark_cluster/build/lib
ln -sf libnccl.so.2.29.7 libnccl.so.2
ln -sf libnccl.so.2 libnccl.so
```

> **Gotcha — there are TWO NCCL code paths.** vLLM's `PyNcclCommunicator` loads
> the patched lib via `VLLM_NCCL_SO_PATH`. But **PyTorch's `torch.distributed`**
> links against torch's bundled NCCL through the pip `nvidia-nccl` package,
> which symlinks to `/usr/lib/aarch64-linux-gnu/libnccl.so.2` (stock, unpatched).
> If you only fix the vLLM path, `determine_available_memory` (a
> `torch.distributed` collective) still hangs on the same timeout. Phase 6's
> `fix_nccl_symlinks` helper handles both — do not skip it.

**Verify Phase 4:** on each node, the patched lib exists and contains the patch marker:

```bash
test -f ~/nccl_spark_cluster/build/lib/libnccl.so.2.29.7 && \
  strings ~/nccl_spark_cluster/build/lib/libnccl.so.2.29.7 | grep -q "Subnet-aware listen" && \
  echo "patch OK" || echo "PATCH MISSING — redo Phase 4"
```

---

## Phase 5 — Get the GLM-5.2 deployment and the model

On the head node, clone MiaAI-Lab's deployment repo:

```bash
git clone https://github.com/MiaAI-Lab/GLM-5.2-NVFP4-AQLM-Triple-DGX-Sparks
cd GLM-5.2-NVFP4-AQLM-Triple-DGX-Sparks
cp .env.example .env     # edit per Phase 7 before serving
```

Bring up the model in stages (these commands come from MiaAI-Lab's `start.sh`):

```bash
./start.sh pull      # distribute the container image to workers (save/rsync/load)
./start.sh download  # fetch the 272 GB model weights to the head
./start.sh sync      # rsync weights to both workers — head + workers MUST match
```

`./start.sh sync` is mandatory. Every node mounts the same checkpoint.

**Verify Phase 5:** `du -sh ~/models/hf/GLM-5.2-NVFP4-AQLM-hybrid` shows ~272 GB
on all three nodes, and the container image is present on all three
(`docker image inspect <image>`).

---

## Phase 6 — Patch `start.sh`: make the NCCL fix durable

`start.sh` launches one Docker container per node. The patched NCCL lib is
**mounted** into each container (read-only) at `/nccl/`, but the container's
*system* NCCL symlinks still point at the stock torch-bundled build. On every
fresh container start you must repoint them — otherwise `torch.distributed` uses
the unpatched lib and the Phase 4 hang returns.

### 6a. Add the helper (after the `vllm_bash()` function)

```bash
# Repoint the container's system + pip nvidia-nccl symlinks to the patched
# NCCL build mounted at NCCL_CONTAINER_DIR. The snippet must contain NO shell
# variables and NO command substitution: it passes through heredoc ->
# remote_on -> printf '%q' -> ssh -> docker -lc, which strips $var forms.
# Stock torch-bundled NCCL hangs on this 3-spark RoCE ring; the custom build
# embeds all PF GIDs in the listen handle (src/transport/net_ib/connect.cc).
fix_nccl_symlinks() {
  local d="${NCCL_CONTAINER_DIR:-/nccl}"
  local tgt="/usr/lib/aarch64-linux-gnu/libnccl.so.2.29.7"
  printf 'cp -n %s/libnccl.so.2.29.7 %s 2>/dev/null; ' "$d" "$tgt"
  printf 'test -f %s || cp -n %s/libnccl.so.2 %s 2>/dev/null; ' "$tgt" "$d" "$tgt"
  printf 'ln -sf libnccl.so.2.29.7 /usr/lib/aarch64-linux-gnu/libnccl.so.2; '
  printf 'ln -sf libnccl.so.2 /usr/lib/aarch64-linux-gnu/libnccl.so; '
}
```

> **Why no shell variables in the snippet?** The worker launch command travels
> `remote_on` → `python3 remote.py` → `printf '%q'` → `ssh` → `docker -lc`.
> That chain strips `$VAR` forms. A snippet with `$L`, `$P`, etc. arrives with
> empty values and silently does nothing. Use only literal paths.

### 6b. Invoke it in the head container launch

Find the head `docker run ... -lc "$(vllm_bash "mkdir -p ...` line and prefix
the `vllm_bash` call (mind the **space** after `)`):

```bash
    -lc "$(fix_nccl_symlinks) $(vllm_bash "mkdir -p '$OBJECT_SPILLING_DIR' && ray start --head \
```

### 6c. Invoke it in the worker container launch

Find the worker `-lc \"cd /opt/vllm && source .venv/bin/activate && mkdir -p ...`
line inside the `remote_on` heredoc and prefix it:

```bash
        -lc \"$(fix_nccl_symlinks) cd /opt/vllm && source .venv/bin/activate && mkdir -p '$OBJECT_SPILLING_DIR' && ray start \
```

### 6d. Make sure the patched lib gets mounted

In `cmd_ray`, replace any single-path NCCL mount with a search:

```bash
      NCCL_VOL=''
      for nccl_dir in $HOME/nccl_spark_cluster/build/lib $HOME/nccl-2.30.7; do
        if [ -f "$nccl_dir/libnccl.so.2" ] || [ -f "$nccl_dir/libnccl.so.2.29.7" ] || [ -f "$nccl_dir/libnccl.so.2.30.7" ]; then
          NCCL_VOL="-v $nccl_dir:$NCCL_CONTAINER_DIR:ro"
          break
        fi
      done
```

### 6e. Thread the new NCCL env vars

`start.sh` must pass `NCCL_NET_MERGE_LEVEL` and `NCCL_DEBUG_FILE` into both
`docker_env_args()` (head) and `worker_docker_env_lines()` (workers). Add them
next to the existing `NCCL_IB_SUBNET_AWARE_ROUTING` lines in both functions:

```bash
    -e "NCCL_NET_MERGE_LEVEL=${NCCL_NET_MERGE_LEVEL:-LOC}"
    -e "NCCL_DEBUG_FILE=/tmp/nccl-log-%h-%p.txt"
```

**Verify Phase 6:** after a `./start.sh serve`, inside any container:

```bash
docker exec <glm52-container> readlink -f /usr/lib/aarch64-linux-gnu/libnccl.so.2
# must resolve to .../libnccl.so.2.29.7  (the patched build), NOT .2.30.7
```

---

## Phase 7 — The `.env` file (recommended config)

This is the **recommended** config: full 380k context, **8 experts** (model
default — full quality), MTP **off** (measured slower on this quantized MoE),
plus the performance recipe from Phase 8. Replace hostnames/user/key with yours.

```bash
# ── Cluster identity (control plane — your Tailscale names here) ──
WORKER_USER=<your-username>
SSH_IDENTITY=$HOME/.ssh/<your-shared-key>
HEAD_HOST=node1
WORKER1_HOST=node2
WORKER2_HOST=node3
GLOO_SOCKET_IFNAME=tailscale0          # control-plane NIC (Tailscale iface)
NCCL_SOCKET_IFNAME=tailscale0

# ── RoCE data plane ──
FABRIC_IFACE=enp1s0f1np1
IB_HCA=rocep1s0f0,rocep1s0f1
NCCL_NET=IB
NCCL_IB_DISABLE=0
NCCL_CROSS_NIC=0
NCCL_IB_MERGE_NICS=0
NCCL_IB_SUBNET_AWARE_ROUTING=1
NCCL_NET_MERGE_LEVEL=LOC
NCCL_IB_GID_INDEX=3
NCCL_GIN_ENABLE=0
NCCL_P2P_DISABLE=1
NCCL_SHM_DISABLE=1
NCCL_GRAPH_MIXING_SUPPORT=1
NCCL_DEBUG=INFO
NCCL_DEBUG_FILE=/tmp/nccl-log-%h-%p.txt
NCCL_DEBUG_SUBSYS=INIT,NET

# ── NCCL small-allreduce tuning (latency-bound decode on a 3-node ring) ──
NCCL_MAX_NCHANNELS=2
NCCL_MIN_NCHANNELS=2
NCCL_BUFFSIZE=4194304
NCCL_PROTO=Simple

# ── NCCL build paths ──
NCCL_HOST_DIR=$HOME/nccl_spark_cluster/build/lib
VLLM_NCCL_SO_PATH=/nccl/libnccl.so.2.29.7

# ── Model / image ──
IMAGE=ghcr.io/miaai-lab/glm-5.2-nvfp4-triple-dgx-sparks:latest
MODEL_DIR=$HOME/models/hf/GLM-5.2-NVFP4-AQLM-hybrid
COMMON_MODEL=/var/tmp/glm52-aqlm

# ── Serving: TP=3, 8 experts (default), MTP OFF, 380k context ──
TP_SIZE=3
DCP_SIZE=1
PP_SIZE=1
PORT=8888
SERVED_MODEL_NAME=glm-5.2
GPU_MEM_UTIL=0.895
MAX_MODEL_LEN=380928
MAX_NUM_SEQS=1
MAX_NUM_BATCHED_TOKENS=4096
KV_CACHE_DTYPE=nvfp4_ds_mla
KV_CACHE_MEMORY_BYTES=12884901888
ENABLE_MTP=0          # MTP measured ~35% slower on this quantized MoE; leave off
ENABLE_DSPARK=0
VLLM_DISABLE_FP8_W8A16=0
# Canonical recipe override (matches MiaAI-Lab README + .env.example). KEEP THIS.
# num_experts_per_tok=8 = the model's trained default (full quality).
# index_topk_freq=8 is a baked, validated value for the :latest build (per the
# repo README, 2026-07-24) — it retunes the sparse-attention indexer, not expert
# count. Do NOT drop this line, and do NOT set num_experts_per_tok=4 unless you
# accept the quality trade (see Phase 8).
HF_OVERRIDES='{"num_experts_per_tok":8,"index_topk_freq":8}'

# ── Performance: FULL CUDA graphs + comm/compute fusion (see Phase 8) ──
CUDAGRAPH_MODE=FULL
CUDAGRAPH_CAPTURE_SIZES=1,2,4,8,12,16,20,24
FUSE_PASSES=ar,norm
FUSE_GEMM_COMMS=0
ASYNC_SCHEDULING=1
VLLM_MARLIN_USE_ATOMIC_ADD=1
```

---

## Phase 8 — Performance tuning (what actually moves the needle)

These changes are **output-quality-neutral**: they change *when* computation
happens (graph capture, comm/compute overlap), not *what* is computed.

| Knob | Stock | Recommended | Why |
|---|---|---|---|
| `CUDAGRAPH_MODE` | `PIECEWISE` | `FULL` | Captures the whole decode step (incl. the all-reduce) as one graph → no per-op launch overhead; comm can overlap compute. |
| `CUDAGRAPH_CAPTURE_SIZES` | `[1,2,4]` | `1,2,4,8,12,16,20,24` | **Must include size 1.** Single-sequence decode runs at batch 1; without a size-1 graph FULL mode falls back to *eager* and gets slower. This is the easy mistake that makes FULL look like a regression. |
| `FUSE_PASSES` | — | `ar,norm` | `ar` = `fuse_allreduce_rms`: fuses the TP all-reduce into the following RMS-norm so comm overlaps compute. Biggest single win on an all-reduce-bound model. |
| `ASYNC_SCHEDULING` | — | `1` | Overlaps scheduling with the engine step. |
| `NCCL_MAX/MIN_NCHANNELS` | 4 | 2 | For the small all-reduces TP decode emits, fewer channels = less per-call setup overhead. Marginal. |

### One honest caveat on reproducibility

`fuse_allreduce_rms` changes the **floating-point reduction order** of the
all-reduce. The math is equivalent, but `temperature=0` (greedy) output is no
longer *bit-for-bit* identical to the unfused path — a single borderline token
can flip, cascading into different (but equally correct) wording. **Quality is
preserved**; only exact byte-reproducibility at temp=0 is affected. If you need
strict reproducibility, drop `ar` from `FUSE_PASSES` (keep `norm`).

### Measured impact

On this cluster (3× GB10, ring RoCE, TP=3, 8 experts, MTP off):

| Config | Throughput | TTFT |
|---|---|---|
| Stock (PIECEWISE, no fusion) | ~7.1 tok/s | ~1.1 s |
| **Recommended (FULL + ar,norm)** | **~8.5 tok/s** | ~1.1 s |

The model is then **engine-bound** (per-token wall-clock ≈ engine iteration
time): there's no further scheduling overhead to remove. Beyond this, the only
levers are quality-affecting (below).

### Optional speed-for-quality dials (NOT in the recommended recipe)

The MiaAI-Lab `.env.example` ships two knobs that trade quality for speed. They
are **independent** of the 380k context setting and of each other.

- **`HF_OVERRIDES='{"num_experts_per_tok":4,...}'`** (the `num_experts_per_tok=4`
  part) — runs **4 active experts per token instead of the model default 8**.
  Halves expert-FFN compute. **Measured gain on this ring: ~8.5 → ~10.0 tok/s
  (+18%)** — not the ~2× the compute math suggests, because this deployment is
  memory-bandwidth-bound (attention, KV reads, the all-reduce, and fixed
  per-step overhead are unchanged). The quality cost is real; see the table
  below. (The `index_topk_freq=8` in that override is a separate, validated
  recipe value for the `:latest` build — keep it either way; it is not the
  quality knob.)

  Expected quality impact on GLM-5.2 (tracks the curve on other modern large
  MoEs that default to top-8, notably the Qwen3 family):

  | Active experts (top-k) | Expected quality vs full GLM-5.2 |
  | --- | --- |
  | 8 (default, recommended) | full strength |
  | 6–7 | mild drop, still very strong |
  | 4 | noticeable-to-significant drop — especially on hard coding, multi-step agent work, long-horizon reasoning |
  | ≤3 | sharp degradation |

  Keep 8 for quality-sensitive work. 4 is defensible for high-throughput chat
  where some loss is acceptable.
- **`ENABLE_MTP=1` + `MTP_SPEC_TOKENS=3`** — Multi-Token Prediction speculative
  decoding. On this NVFP4/AQLM-quantized MoE we **measured MTP ~35% slower**
  than off (draft rejection costs more than it saves for diverse prompts). It
  can help on highly predictable text.

---

## Phase 9 — Bring it up

```bash
./start.sh serve      # Ray (3 nodes, 3 GPUs) + vLLM serve on :8888
```

Watch for `Application startup complete` in the serve log
(`logs/glm52-aqlm.log`, or `/tmp/vllm-serve.log` inside the head container).
First load takes several minutes (272 GB across 3 nodes + CUDA-graph capture).

Smoke test:

```bash
curl -s http://localhost:8888/v1/models | head -c 200
curl -s http://localhost:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"Say hi"}],"max_tokens":20}' \
  | head -c 400
```

Benchmark (streaming, single-stream, end-to-end):

```bash
python3 - <<'PY'
import requests, time, json
url = "http://localhost:8888/v1/completions"
requests.post(url, json={"model":"glm-5.2","prompt":"hi","max_tokens":5,"temperature":0}, timeout=120)  # warmup
payload = {"model":"glm-5.2","prompt":"Write a long detailed essay about the history of computing.","max_tokens":400,"temperature":0.0,"stream":True}
t0=time.time(); first=None; n=0
with requests.post(url, json=payload, stream=True, timeout=300) as r:
    for line in r.iter_lines():
        if not line: continue
        l=line.decode("utf-8","ignore")
        if l.startswith("data: ") and "[DONE]" not in l:
            if first is None: first=time.time()-t0
            try:
                o=json.loads(l[6:])
                if o.get("choices") and o["choices"][0].get("text"): n+=1
            except: pass
tot=time.time()-t0; gen=tot-(first or tot)
print(f"{n} tok | TTFT {first:.2f}s | {n/gen:.2f} tok/s")
PY
```

Stop / restart:

```bash
./start.sh stop
./start.sh serve
```

---

## Troubleshooting — the failure modes you will hit

### `ibv_modify_qp failed with 110 Connection timed out`

The signature symptom of the ring-routing problem. Causes, in likelihood order:

1. **Phase 4's NCCL patch isn't loaded by `torch.distributed`.** Confirm inside a
   container:
   ```bash
   docker exec <glm52-container> readlink -f /usr/lib/aarch64-linux-gnu/libnccl.so.2
   # must resolve to .../libnccl.so.2.29.7  (the patched build)
   strings <that-path> | grep "Subnet-aware listen"   # must print the marker
   ```
   If it resolves to `libnccl.so.2.30.7`, Phase 6's `fix_nccl_symlinks` isn't
   running — recheck 6b/6c.
2. **Subnet/GID-index mismatch.** `NCCL_IB_GID_INDEX` must point at the RoCEv2
   IPv4 GID on every port on every node (index 3 on the stock Spark layout).
   Re-run Phase 3.
3. **A node can't reach a peer's RoCE subnet at all** (e.g. `ip route get
   192.168.4.1` on node1 exits via WiFi/LAN, not a CX-7 port). That's a
   netplan/topology error — recheck Phase 1. RoCE here is point-to-point; there
   is no forwarding between links.

### Worker containers exit immediately (exit 127, `:cd: command not found`)

A shell snippet with `$VAR` forms got mangled by the `remote_on` → `printf '%q'`
→ ssh → `docker -lc` escape chain. `fix_nccl_symlinks` must emit **literal paths
only** (no `$L`, `$(...)`, backticks). See the warning in 6a. Also ensure a
**space** after `$(fix_nccl_symlinks)` in both injection points.

### `Ray did not reach 3 GPUs`

A worker container failed to start or Ray couldn't reach it. Check:
```bash
ssh <worker> docker logs glm52-aqlm-worker
ssh <worker> docker exec glm52-aqlm-worker readlink -f /usr/lib/aarch64-linux-gnu/libnccl.so.2
```
Some forks of `stop.sh` reference stale static worker IPs — if it can't reach
workers to clean up, remove them manually:
```bash
ssh <worker> docker rm -f glm52-aqlm-worker
```
then re-run `./start.sh serve`.

### FULL graph mode made it *slower* (eager fallback)

You're missing a **size-1** capture. With `MAX_NUM_SEQS=1`, single-stream decode
runs at batch 1; if `1` isn't in `CUDAGRAPH_CAPTURE_SIZES`, FULL mode has no
matching graph and falls back to slow eager execution. Always include `1` (see
Phase 7/8). PIECEWISE mode ignores `cudagraph_capture_sizes` entirely (vLLM
fills defaults), which is why it "just worked" before.

### `shm_broadcast: No available shared memory broadcast block found in 60 seconds`

**Usually normal during load.** It means workers are busy with compilation /
weight loading / KV-cache quantization, not dead. Only worry if it persists past
`Application startup complete` or accompanies an `ibv_modify_qp` error.

### NCCL version mismatch in the error (`NCCL version 2.30.7` in a vLLM traceback)

Confirms `torch.distributed` is loading the **stock** lib, not your patched
build. This is exactly the two-code-path gotcha from Phase 4 — apply Phase 6.

---

## Summary of every change from stock

1. **NCCL source** (`src/transport/net_ib/connect.cc`, `ncclIbListen`): embed all
   PF GIDs in the listen handle. Rebuild → `libnccl.so.2.29.7` on all 3 nodes. (Phase 4)
2. **`start.sh`**: add `fix_nccl_symlinks()`, call it in head + worker launches,
   widen the `NCCL_VOL` mount search, thread `NCCL_NET_MERGE_LEVEL` /
   `NCCL_DEBUG_FILE` into both env functions. (Phase 6)
3. **`.env`**: the Phase 7 recipe — 8 experts (canonical `HF_OVERRIDES`), MTP off,
   380k context, FULL graphs + `ar,norm` fusion + NCCL latency tuning.

The model, the vLLM image, and MiaAI-Lab's `start.sh` structure are used as-is;
only the integration glue above is added.
