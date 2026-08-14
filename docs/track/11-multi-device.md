---
title: "11 · Multi-device"
parent: "C++ ↔ CUDA Track"
nav_order: 11
permalink: /docs/track/11-multi-device/
---

# 11 · Multi-device
{: .no_toc }

**Stage D** · ⏳ ~4h · prerequisites: [module 10](../10-advanced-gpu/) · [lab source](https://github.com/trocca/ai-infra-gpu-engineering-ramp/blob/main/cpp-cuda-track/11-multi-device)

---

Scaling past one device — where this track merges back into the main
PyTorch/parallelism ramp. The concepts here *are* the transport layer under
DDP, FSDP, and tensor parallelism.

## Concept

**CPU scaling:** first NUMA (two+ sockets, each with local DRAM; remote access
~1.5–2× slower — `numactl`, first-touch policy), then **MPI** across machines:
explicit ranks, explicit messages, and *collectives* — `MPI_Allreduce`,
`MPI_Allgather`, `MPI_ReduceScatter`. Forty years of HPC wisdom in one API.

**GPU scaling:** same shape, denser fabric. Within a node: **NVLink/NVSwitch**
(900 GB/s/GPU on H100, ~1.8 TB/s on Blackwell NVL72 — vs PCIe5's ~64 GB/s),
peer-to-peer access (`cudaMemcpyPeer`, or direct load/store). Across nodes:
InfiniBand/RoCE with **GPUDirect RDMA** (NIC reads GPU memory, CPU never
touches the bytes). And the API everyone actually uses: **NCCL** — MPI's
collectives, reimplemented to run *as GPU kernels* over these fabrics, with
ring and tree algorithms chosen automatically.

The Rosetta stone (this table is half of distributed ML):

| Collective | What it does | Who uses it |
|---|---|---|
| all-reduce | every rank ends with the global sum | DDP gradient sync |
| reduce-scatter | global sum, sharded across ranks | FSDP backward |
| all-gather | every rank gets all shards | FSDP forward (params) |
| broadcast | rank 0 → everyone | initial weights |
| all-to-all | personalized exchange | MoE token routing |

Cost model to memorize: ring all-reduce moves `2·(n-1)/n · bytes` per rank —
nearly bandwidth-optimal, latency-poor at small sizes (hence tree algorithms
and gradient bucketing in DDP).

## Confrontation

| Question | CPU stack | GPU stack |
|---|---|---|
| Local vs remote memory | NUMA node | HBM vs peer GPU vs host |
| Fabric | UPI / Ethernet / IB | NVLink / NVSwitch / IB+GPUDirect |
| Collectives API | MPI | NCCL (MPI-shaped, GPU-resident) |
| Rank bootstrap | `mpirun` | `torchrun` env / MPI / NCCL id exchange |
| Overlap comm & compute | MPI_I* + threads | NCCL on its own stream + events |

## Exercises

1. CPU: `numactl --membind` a bandwidth benchmark local vs remote; measure the
   NUMA penalty. GPU: `cudaMemcpyPeer` bandwidth between two GPUs with and
   without `cudaDeviceEnablePeerAccess` (through-host vs NVLink path);
   `nvidia-smi topo -m` first to know your topology.
2. Write ring all-reduce by hand twice: MPI point-to-point on CPU, then
   two-GPU with peer copies. Verify both against the library one-liner
   (`MPI_Allreduce`, `ncclAllReduce`). Nothing teaches the cost model better.
3. `nccl-tests` (`all_reduce_perf -b 1K -e 1G`): plot bus bandwidth vs message
   size; find where latency-bound becomes bandwidth-bound. Connect to DDP's
   `bucket_cap_mb` default (25 MB) — now you know why it exists.
4. Run a 2-GPU PyTorch DDP script under `nsys`; find the NCCL all-reduce
   kernels overlapping the backward pass. This is the track's graduation
   photo: module 01's warps to module 11's fleet, one timeline.

## Lab

- [`cpp/ring_allreduce.cpp`](https://github.com/trocca/ai-infra-gpu-engineering-ramp/blob/main/cpp-cuda-track/11-multi-device/cpp/ring_allreduce.cpp) — exercise 2 (ring all-reduce by hand: reduce-scatter + all-gather, threads standing in for ranks; verifies the 2·(R−1)/R traffic formula). Build: `g++ -O3 -std=c++20 -pthread cpp/ring_allreduce.cpp -o ring_cpu`
- [`cuda/p2p_bandwidth.cu`](https://github.com/trocca/ai-infra-gpu-engineering-ramp/blob/main/cpp-cuda-track/11-multi-device/cuda/p2p_bandwidth.cu) — exercise 1 (pairwise GPU bandwidth matrix, peer access off/on; single-GPU fallback included). Build: `nvcc -O3 -arch=native cuda/p2p_bandwidth.cu -o p2p_gpu`

## Companion reading

- Fregly, *AI Systems Performance Engineering*: Ch. 2 §"Ultra-Scale Networking, NVLink and NVSwitch", Ch. 3 §"NUMA Awareness and CPU Pinning", Ch. 4 (NCCL, topology awareness, comm/compute overlap, SHARP — this module's syllabus as a chapter), Ch. 12 §"NVSHMEM".
- Ultra-Scale Playbook: data-parallelism + ZeRO sections — the training-strategy view of these same collectives.

---

Next: [12 · Capstone: PyTorch extension](../12-capstone-pytorch-extension/)
