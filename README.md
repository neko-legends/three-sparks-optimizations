# GLM-5.2 across three DGX Sparks in a RoCE ring

This repository makes **GLM-5.2 (NVFP4+AQLM hybrid)** run across three NVIDIA DGX
Spark nodes wired in a point-to-point RoCE ring, served with vLLM at
tensor-parallel=3 — and documents every non-obvious fix required to get there,
because the stock instructions do not complete on this topology.

It builds on two open-source projects:

- **NVIDIA's [`dgx-spark-playbooks`](https://github.com/NVIDIA/dgx-spark-playbooks)**
  *"Connect Three DGX Spark in a Ring Topology"* playbook — the physical wiring,
  netplan, and NCCL sanity-test foundation (Phase 1 below).
- **[`MiaAI-Lab/GLM-5.2-NVFP4-AQLM-Triple-DGX-Sparks`](https://github.com/MiaAI-Lab/GLM-5.2-NVFP4-AQLM-Triple-DGX-Sparks)**
  — the GLM-5.2 deployment scripts, vLLM fork, container image, and the model
  weights themselves.

Neither is replaced here. NVIDIA's playbook gets you a working ring; MiaAI-Lab's
repo gets you the model. **What this repo adds is the bridge between them**: the
patches and config that make MiaAI-Lab's `./start.sh serve` actually complete on
the ring NVIDIA's playbook produces.

## Why this repository exists

MiaAI-Lab's stock deployment hangs at NCCL init on the three-Spark ring. The
root cause is subtle and worth stating plainly:

> NCCL's graph layer assigns communication channels to NICs round-robin per
> peer. On this ring each peer is reachable on only **one** subnet, so half the
> channels get told to use a NIC that physically cannot reach the peer — and
> `ibv_modify_qp` times out (`110 Connection timed out`). The custom NCCL build's
> subnet-aware routing *should* fix this, but its listener only advertised its
> own single-subnet GID, so the connector had no reachable GID to match.

This repo provides a **6-line patch to `ncclIbListen`** that makes the listener
advertise all of its RoCE interfaces, so the connector can always find a
mutually-reachable subnet. Plus the surrounding glue (a `start.sh` helper that
makes the patched NCCL win against torch's bundled copy on every container
start) and a measured performance recipe.

The result is a serving GLM-5.2 with full 8-expert quality and the model's
intended 380k-token context.

**Measured** (3× GB10, ring RoCE, TP=3, 8 experts, MTP off, 380k context):

| Config | Throughput (single stream) | Notes |
| --- | --- | --- |
| MiaAI-Lab stock (PIECEWISE graphs, no fusion) | ~7.1 tok/s | hangs unless this repo's NCCL patch is applied |
| + FULL CUDA graphs + all-reduce/RMS-norm fusion | ~7.8 tok/s | quality-neutral: same math, scheduled differently |
| **+ MTP off (recommended, 8 experts)** | **~8.5 tok/s** | full model quality; MTP measured slower on this quantized MoE |
| + drop to 4 experts (`num_experts_per_tok=4`) | ~10.0 tok/s | **+18%, not 2×** — see the quality trade below |

### The 4-expert trade, measured honestly

Halving active experts (8→4) sounds like it should roughly double speed, since
it halves expert-FFN compute. **It doesn't — this deployment is
memory-bandwidth-bound, not compute-bound**, so attention, KV-cache reads, the
TP all-reduce, and fixed per-step overhead are unchanged. The real measured gain
is **~18% (8.5 → 10.0 tok/s)**, useful but not transformative.

The cost is quality. Expected impact on GLM-5.2 tracks the curve seen on other
modern large MoEs that default to top-8 (notably the Qwen3 family):

| Active experts (top-k) | Expected quality vs full GLM-5.2 |
| --- | --- |
| 8 (default, recommended) | full strength |
| 6–7 | mild drop, still very strong |
| 4 | noticeable-to-significant drop — especially on hard coding, multi-step agent work, and long-horizon reasoning |
| ≤3 | sharp degradation |

**Recommendation:** keep 8 experts for quality-sensitive work (coding, agents,
reasoning). 4 experts is a defensible choice for high-throughput chat where some
quality loss is acceptable — but go in with eyes open about the trade, and do
not expect the 2× speedup the compute math suggests.

Single-stream tok/s is the latency metric. **Aggregate throughput under
concurrent load is much higher** — batched decode shares the fixed per-token
overhead across requests, so real multi-request workloads see far more than
8.5 tok/s.

## What's in here

- **[`SETUP.md`](SETUP.md)** — the full, reproducible, agent-followable guide.
  Phased: hardware ring → Tailscale control plane → NCCL patch → model →
  serving → tuning → troubleshooting. Every failure mode we actually hit is in
  the troubleshooting section with its exact error string and fix.
- The **NCCL patch** is inline in SETUP.md §3 and reproducible from source.
- **No model weights, no container image, no proprietary code** is redistributed
  here — only configs, patches, and instructions. You fetch the model and image
  from their original sources.

## For AI agents

This repository is structured to be followable by a coding agent tasked with
*"install GLM-5.2 on my three Sparks."* [`SETUP.md`](SETUP.md) opens with a
prerequisites checklist and a per-phase verification flow, so an agent can
diagnose the cluster state, run each phase, and verify success before moving on.
The troubleshooting section maps exact error strings to fixes.

**Agent quick-start:** read [`SETUP.md`](SETUP.md) §0 (prerequisites) and §1
(diagnostic checklist), then execute phases in order. Do not skip the NCCL patch
in §3 — the deployment will hang without it, and the hang is silent (a 2-minute
timeout per QP, not a crash).

## License & credit

The NCCL patch in §3 modifies [NVIDIA's NCCL contrib tree](https://github.com/NVIDIA/nccl)
(commit `fab1850`, *"Support 3 DGX Sparks in a ring topology via CX-7"*), which
is itself BSD-licensed. The deployment scripts and vLLM fork are from
[MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.2-NVFP4-AQLM-Triple-DGX-Sparks).
The physical-ring foundation is from
[NVIDIA's `dgx-spark-playbooks`](https://github.com/NVIDIA/dgx-spark-playbooks).
This repository contributes only the integration patches and configuration
documented in `SETUP.md`.

See [`SETUP.md`](SETUP.md) for the step-by-step.
