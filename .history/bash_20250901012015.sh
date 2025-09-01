cat > nccl_smoke.py <<'PY'
import os, torch, torch.distributed as dist
def main(rank, world=2):
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    t = torch.ones(4, device=f"cuda:{rank}")
    dist.all_reduce(t)
    print(f"[rank{rank}] allreduce OK:", t.tolist())
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    # torchrun이 넘겨주는 환경변수 사용
    r = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "2"))
    main(r, world)
PY