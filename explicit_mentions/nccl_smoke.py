"""Minimal multi-GPU NCCL check -- reproduces (or clears) the hang that killed
train_708003 without touching the model, the data, or OLMo-core.

The failure there was `WorkNCCL(SeqNum=1, OpType=BROADCAST, NumelIn=16)` timing
out after 30 minutes: the first collective never completed on some ranks. This
script performs exactly that first handshake, with a short timeout, so the
answer arrives in seconds instead of half an hour.

Run under torchrun with the same GPU count as training:
    torchrun --nproc-per-node=4 nccl_smoke.py

If it hangs or times out, the problem is the NCCL/GPU environment on that node,
not our training code. Retry with peer-to-peer transport disabled:
    NCCL_P2P_DISABLE=1 torchrun --nproc-per-node=4 nccl_smoke.py
and with more detail:
    NCCL_DEBUG=INFO torchrun --nproc-per-node=4 nccl_smoke.py
"""
import datetime
import os
import sys

import torch
import torch.distributed as dist


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    # 3 minutes, not the 30-minute default: fail fast.
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        timeout=datetime.timedelta(minutes=3),
    )
    rank, world = dist.get_rank(), dist.get_world_size()
    name = torch.cuda.get_device_name(local_rank)
    cap = torch.cuda.get_device_capability(local_rank)
    print(f"[rank {rank}/{world}] local_rank={local_rank} {name} sm_{cap[0]}{cap[1]} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

    print(f"[rank {rank}] gloo barrier...", flush=True)
    dist.barrier()                        # CPU-only: isolates NCCL from rendezvous
    print(f"[rank {rank}] gloo barrier OK", flush=True)

    # The exact op that hung: a tiny broadcast on the GPU.
    print(f"[rank {rank}] nccl broadcast (16 elems)...", flush=True)
    t = torch.full((16,), float(rank), device="cuda")
    dist.broadcast(t, src=0)
    torch.cuda.synchronize()
    assert t.tolist() == [0.0] * 16, t.tolist()
    print(f"[rank {rank}] nccl broadcast OK", flush=True)

    print(f"[rank {rank}] nccl all_reduce (1M elems)...", flush=True)
    big = torch.ones(1 << 20, device="cuda")
    dist.all_reduce(big)
    torch.cuda.synchronize()
    assert big[0].item() == world, big[0].item()
    print(f"[rank {rank}] nccl all_reduce OK (sum={big[0].item()})", flush=True)

    # Peer-to-peer reachability, which is what usually breaks on shared nodes.
    if rank == 0:
        print("\n[rank 0] peer-to-peer access matrix (visible devices):")
        n = torch.cuda.device_count()
        for i in range(n):
            row = [("yes" if torch.cuda.can_device_access_peer(i, j) else "NO ")
                   if i != j else " - " for j in range(n)]
            print(f"    gpu{i}: {' '.join(row)}")
        print(f"\n[rank 0] NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE', 'unset')}"
              f"  NCCL_SHM_DISABLE={os.environ.get('NCCL_SHM_DISABLE', 'unset')}")

    dist.barrier()
    if rank == 0:
        print("\nALL NCCL CHECKS PASSED", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
